"""Training/evaluation helpers; importing this module never starts a run."""
from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .fusion_data import make_collator, make_dataset, repo_path


def read_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    for name in ("batch_size", "epochs", "patience"):
        if not isinstance(t[name], int) or t[name] < 1:
            raise ValueError(f"train.{name} must be a positive integer")
    if not 0 <= t["freeze_epochs"] <= t["epochs"] or not 0 <= t["unfreeze_last"] <= 6:
        raise ValueError("Invalid freeze_epochs or unfreeze_last")
    if not 1 <= d["max_length"] <= 512 or d["workers"] < 0:
        raise ValueError("Invalid max_length or workers")
    if m["hidden_dim"] < 1 or not 0 <= m["dropout"] < 1:
        raise ValueError("Invalid fusion head dimensions/dropout")
    for name in ("lr_head", "lr_encoder", "grad_clip"):
        if float(t[name]) <= 0:
            raise ValueError(f"train.{name} must be positive")
    if float(t["weight_decay"]) < 0:
        raise ValueError("weight_decay must be non-negative")
    return cfg


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(_):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def choose_device(value="auto"):
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(batch, device):
    return {k: v.to(device, non_blocking=device.type == "cuda") if torch.is_tensor(v) else v
            for k, v in batch.items()}


def make_loader(cfg, rows, tokenizer, device, *, train=False):
    return DataLoader(
        make_dataset(cfg, rows, train=train), batch_size=cfg["train"]["batch_size"],
        shuffle=train, num_workers=cfg["data"]["workers"],
        collate_fn=make_collator(cfg, tokenizer), pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(cfg["seed"]),
    )


def classification_metrics(confusion):
    cm = confusion.double()
    tp = cm.diag()
    support, predicted = cm.sum(1), cm.sum(0)
    precision = tp / predicted.clamp_min(1)
    recall = tp / support.clamp_min(1)
    f1 = 2 * tp / (support + predicted).clamp_min(1)
    return {"accuracy": float(tp.sum() / cm.sum().clamp_min(1)),
            "macro_f1": float(f1.mean()), "samples": int(cm.sum()),
            "per_class": [{"label": i, "precision": float(precision[i]),
                           "recall": float(recall[i]), "f1": float(f1[i]),
                           "support": int(support[i])} for i in range(len(cm))],
            "confusion_matrix": confusion.tolist()}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def run_epoch(model, loader, device, *, optimizer=None, scaler=None,
              scheduler=None, amp=False, grad_clip=1.0, progress_every=0):
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    confusion = torch.zeros(12, 12, dtype=torch.long)
    predictions, loss_sum, count = [], 0.0, 0
    enabled = amp and device.type == "cuda"
    if training and scaler is None:
        raise ValueError("Training requires a GradScaler (it may be disabled on CPU)")
    synchronize(device)
    start = time.perf_counter()
    with torch.set_grad_enabled(training):
        for step, raw_batch in enumerate(loader, 1):
            batch = to_device(raw_batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled):
                logits = model(batch)
                loss = criterion(logits, batch["labels"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite fusion loss")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # With AMP, GradScaler must be allowed to skip/back off on overflow.
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=not enabled)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None and scaler.get_scale() >= previous_scale:
                    scheduler.step()
            probability, predicted = logits.detach().float().softmax(1).max(1)
            y, p = batch["labels"].detach().cpu(), predicted.cpu()
            confusion += torch.bincount(y * 12 + p, minlength=144).reshape(12, 12)
            count += len(y)
            loss_sum += float(loss.detach()) * len(y)
            if not training:
                predictions += [{"item_id": item, "label": int(label), "prediction": int(pred),
                                 "confidence": float(prob)} for item, label, pred, prob in
                                zip(raw_batch["item_id"], y, p, probability.cpu())]
            if progress_every and (step % progress_every == 0 or step == len(loader)):
                elapsed = time.perf_counter() - start
                print(f"{'Train' if training else 'Validate'} {step}/{len(loader)} | "
                      f"loss {loss_sum / count:.4f} | {count / max(elapsed, 1e-6):.1f} pairs/s", flush=True)
    if not count:
        raise ValueError("Empty data loader")
    synchronize(device)
    metrics = classification_metrics(confusion)
    metrics.update(loss=loss_sum / count, seconds=time.perf_counter() - start)
    return metrics, predictions


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path, model, cfg, labels, fingerprint, *, epoch, best_val_f1):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "config": cfg, "labels": labels,
                "fingerprint": fingerprint, "text_config": model.text_encoder.bert.config.to_dict(),
                "epoch": epoch, "best_val_f1": best_val_f1}, temp)
    temp.replace(path)


def load_checkpoint(path, *, data_dir=None):
    from .fusion_data import split_fingerprint
    from .models.fusion import build_fusion

    ck = torch.load(path, map_location="cpu", weights_only=True)
    folder = data_dir if data_dir is not None else ck["config"]["data"]["dir"]
    if ck["fingerprint"] != split_fingerprint(folder):
        raise ValueError("Checkpoint and split manifests/labels.json do not match")
    labels = json.loads((repo_path(folder) / "labels.json").read_text(encoding="utf-8"))
    if ck["labels"] != labels:
        raise ValueError("Checkpoint label mapping does not match labels.json")
    model = build_fusion(ck["config"], pretrained=False, text_config=ck["text_config"])
    model.load_state_dict(ck["model"], strict=True)
    return model, ck
