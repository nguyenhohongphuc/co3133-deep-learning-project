"""Vòng train A2: giống src/engine/trainer.fit của A1, chỉ khác optimizer có hai nhóm tham số.

Thí nghiệm có kiểm soát frozen vs full fine-tune (proposal mục 9): hai nhánh dùng cùng lr cho head
(training.lr) và chỉ khác lr của backbone (training.lr_backbone): 0 = đóng băng, 1e-4 = fine-tune.
A1 dựng optimizer từ model.parameters() với một lr duy nhất nên không làm được việc này;
file A1 giữ nguyên, ở đây chỉ import lại các hàm của nó.
"""
from __future__ import annotations

import pandas as pd
import torch

from src.engine.trainer import accuracy, build_loss, evaluate, macro_f1, resolve_device, train_one_epoch


def build_param_groups(model, training_config: dict) -> list[dict]:
    """Head (tham số có tên bắt đầu bằng model.head_prefix) dùng lr; phần còn lại dùng lr_backbone.

    Model không có head_prefix (CNN baseline) -> một nhóm duy nhất với lr, như A1.
    Tham số requires_grad=False (backbone bị đóng băng) không được đưa vào optimizer.
    """
    lr = training_config["lr"]
    head_prefix = getattr(model, "head_prefix", None)
    if head_prefix is None:
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]

    head, backbone = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (head if name.startswith(head_prefix) else backbone).append(p)
    groups = [{"params": head, "lr": lr}]
    lr_backbone = training_config.get("lr_backbone", lr)
    if backbone:
        if lr_backbone <= 0:
            raise ValueError("lr_backbone <= 0 but the backbone is trainable: set freeze_backbone: true")
        groups.append({"params": backbone, "lr": lr_backbone})
    elif lr_backbone > 0:
        raise ValueError("lr_backbone > 0 but the backbone is frozen: set lr_backbone: 0")
    return groups


def build_optimizer(model, config: dict):
    training_config = config["training"]
    groups = build_param_groups(model, training_config)
    weight_decay = training_config.get("weight_decay", 0.0)
    if training_config["optimizer"] == "adam":
        return torch.optim.Adam(groups, weight_decay=weight_decay)
    if training_config["optimizer"] == "sgd":
        return torch.optim.SGD(groups, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {training_config['optimizer']}")


def fit(model, loaders, config, run_dir):
    device = resolve_device(config)
    model = model.to(device)
    loss_fn = build_loss(config)
    optimizer = build_optimizer(model, config)
    monitor = config["training"]["monitor"]
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    history = []
    best_epoch = -1
    best_value = None
    wait = 0
    for epoch in range(config["training"]["epochs"]):
        train_loss = train_one_epoch(model, loaders["train"], loss_fn, optimizer, device)
        val_loss, y_true, y_pred = evaluate(model, loaders["val"], loss_fn, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": accuracy(y_true, y_pred),
            "val_macro_f1": macro_f1(y_true, y_pred),
        }
        history.append(row)
        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"val_accuracy={row['val_accuracy']:.4f}, val_macro_f1={row['val_macro_f1']:.4f}")
        if best_value is None or row[monitor] > best_value:
            best_value = row[monitor]
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            wait += 1
            if wait >= config["training"]["patience"]:
                print(f"Early stopping at epoch {epoch} due to no improvement in {monitor} for {wait} epochs.")
                break
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    peak = torch.cuda.max_memory_allocated() / 2**30 if str(device).startswith("cuda") else None
    return {"history": history, "best_epoch": best_epoch, "best_value": best_value, "device": device,
            "peak_gpu_mem_gb": peak}
