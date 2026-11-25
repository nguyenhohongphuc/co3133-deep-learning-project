"""Train concat fusion. Test evaluation is a separate, explicit command."""
from __future__ import annotations

import argparse
import importlib.metadata
import hashlib
import subprocess
import time

import torch
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from src.a3.fusion_data import load_fusion_splits, repo_path, split_fingerprint
from src.a3.fusion_engine import (choose_device, make_loader, read_config, run_epoch,
                                  save_checkpoint, seed_all, write_csv, write_json)
from src.a3.models.fusion import build_fusion


def train(cfg, device_name="auto"):
    seed_all(cfg["seed"])
    out = repo_path(cfg["output_dir"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {out}; choose a new output_dir")
    splits, labels = load_fusion_splits(cfg)
    device = choose_device(device_name)
    print(f"Device: {device} | split counts: { {k: len(v) for k, v in splits.items()} }", flush=True)
    settings, m = cfg["train"], cfg["model"]
    print("Loading pretrained ResNet50 and DistilBERT...", flush=True)
    model = build_fusion(cfg).to(device)
    # Keep tokenizer and encoder on the same resolved Hub revision when available.
    resolved_revision = getattr(model.text_encoder.bert.config, "_commit_hash", None)
    tokenizer = AutoTokenizer.from_pretrained(m["name"], revision=resolved_revision or m.get("revision"))
    if cfg["data"]["max_length"] > model.text_encoder.bert.config.max_position_embeddings:
        raise ValueError("max_length exceeds the text encoder's positional capacity")
    train_loader = make_loader(cfg, splits["train"], tokenizer, device, train=True)
    val_loader = make_loader(cfg, splits["validation"], tokenizer, device)
    optimizer = torch.optim.AdamW([
        {"params": model.image_encoder.parameters(), "lr": float(settings["lr_encoder"])},
        {"params": model.text_encoder.parameters(), "lr": float(settings["lr_encoder"])},
        {"params": model.head.parameters(), "lr": float(settings["lr_head"])},
    ], weight_decay=float(settings["weight_decay"]))
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, settings["epochs"] * len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=settings["amp"] and device.type == "cuda")
    out.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(out / "tokenizer")
    fingerprint = split_fingerprint(cfg["data"]["dir"])
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_path("."), text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    versions = {p: importlib.metadata.version(p) for p in
                ("torch", "torchvision", "transformers", "numpy", "Pillow", "PyYAML")}
    # Capture transitive dependencies without changing the user's environment.
    (out / "requirements.lock.txt").write_text("\n".join(sorted(
        f"{dist.metadata['Name']}=={dist.version}" for dist in importlib.metadata.distributions()
        if dist.metadata["Name"])) + "\n", encoding="utf-8")
    write_json(out / "run_config.json", {
        "config": cfg, "labels": labels, "fingerprint": fingerprint, "git_commit": commit,
        "versions": versions, "device": str(device), "resolved_text_revision": resolved_revision,
        "image_weights": m["image_weights"], "parameters": sum(p.numel() for p in model.parameters()),
        "pooling": "attention-masked mean (including valid special tokens), shared TextEncoder",
        "image_preprocessing_source": "sonphan:68e9f7e/src/a3_image_only/dataset.py",
        "split_counts": {k: len(v) for k, v in splits.items()},
        "source_sha256": {str(p.relative_to(repo_path('.'))): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [repo_path('scripts/a3_train_fusion.py'),
                                    repo_path('src/a3/fusion_engine.py'), repo_path('src/a3/fusion_data.py'),
                                    repo_path('src/a3/models/fusion.py'), repo_path('src/a3/models/text_only.py'),
                                    repo_path('src/a3/data.py')]},
    })
    best, bad_epochs, history = -1.0, 0, []
    start = time.perf_counter()
    for epoch in range(1, settings["epochs"] + 1):
        fine_tune = epoch > settings["freeze_epochs"]
        model.configure_finetuning(fine_tune, settings["unfreeze_last"] if fine_tune else 0)
        print(f"Starting epoch {epoch}/{settings['epochs']} ({'finetune' if fine_tune else 'head-only'})", flush=True)
        tr, _ = run_epoch(model, train_loader, device, optimizer=optimizer, scaler=scaler,
                          scheduler=scheduler, amp=settings["amp"], grad_clip=settings["grad_clip"],
                          progress_every=25)
        va, _ = run_epoch(model, val_loader, device, amp=settings["amp"], progress_every=25)
        history.append({"epoch": epoch, "stage": "finetune" if fine_tune else "head",
                        **{f"train_{k}": tr[k] for k in ("loss", "accuracy", "macro_f1", "seconds")},
                        **{f"val_{k}": va[k] for k in ("loss", "accuracy", "macro_f1", "seconds")}})
        write_csv(out / "history.csv", history)
        if va["macro_f1"] > best:
            best, bad_epochs = va["macro_f1"], 0
            save_checkpoint(out / "best.pt", model, cfg, labels, fingerprint,
                            epoch=epoch, best_val_f1=best)
            write_json(out / "best_validation_metrics.json", va)
        else:
            bad_epochs += 1
        save_checkpoint(out / "last.pt", model, cfg, labels, fingerprint,
                        epoch=epoch, best_val_f1=best)
        print(f"Epoch {epoch}/{settings['epochs']} | loss {tr['loss']:.4f} | "
              f"val macro-F1 {va['macro_f1']:.4f} | best {best:.4f}", flush=True)
        if bad_epochs >= settings["patience"]:
            break
    write_json(out / "training_summary.json", {"best_val_macro_f1": best,
               "epochs_completed": len(history), "training_seconds": time.perf_counter() - start})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="configs/a3_fusion.yaml")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0 or mps")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    cfg = read_config(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    print(f"Saved run: {train(cfg, args.device)}")


if __name__ == "__main__":
    main()
