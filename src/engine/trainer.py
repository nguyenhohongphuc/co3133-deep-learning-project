"""Train/validation loop. Owner: TV 2. Task A1-02.

Handbook 3.2 wants all of this recorded: optimizer, lr, batch size, epochs, scheduler,
regularization, early stopping, checkpoint criterion, seed, hardware, training time.

CONTRACT (agreed 20/09):
    fit(model, loaders, config, run_dir) -> dict with the training history
    - writes history.csv (one row per epoch) into run_dir
    - saves best.pt whenever val_macro_f1 improves (config['training']['checkpoint_metric'])

best.pt holds {"epoch", "model_state"}; load it with
    model.load_state_dict(torch.load(path)["model_state"])
"""
from __future__ import annotations

import csv
import pathlib
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from src.engine.metrics import accuracy, macro_f1

    
def build_loss(config):
    loss_name = config["training"]["loss"]
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_name == "mse":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

def build_optimizer(model, config):
    optimizer_name = config["training"]["optimizer"]
    # weight_decay is the L2 regularization the report has to state (handbook 3.2).
    weight_decay = config["training"].get("weight_decay", 0.0)
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config["training"]["lr"],
                               weight_decay=weight_decay)
    elif optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=config["training"]["lr"],
                                weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def resolve_device(config) -> str:
    """config['training']['device']: 'auto' | 'cpu' | 'cuda'."""
    wanted = config["training"].get("device", "auto")
    if wanted == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if wanted == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config asks for cuda but torch.cuda.is_available() is False")
    return wanted

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    
    model.train()
    total_loss = 0.0
    total_sample = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_sample += images.size(0)
    return total_loss / total_sample



def evaluate(model, loader, loss_fn, device):
    """Returns (mean loss, y_true, y_pred); the labels feed the epoch metrics."""
    model.eval()
    total_loss = 0.0
    total_sample = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item() * images.size(0)
            total_sample += images.size(0)
            y_true.append(labels.cpu().numpy())
            y_pred.append(outputs.argmax(dim=1).cpu().numpy())
    return total_loss / total_sample, np.concatenate(y_true), np.concatenate(y_pred)


def fit(model, loaders, config, run_dir):
    device = resolve_device(config)
    model = model.to(device)
    loss_fn = build_loss(config)
    optimizer = build_optimizer(model, config)

    run_dir = pathlib.Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_value = None
    best_epoch = -1
    bad_epochs = 0
    patience = config["training"]["early_stopping_patience"]

    for epoch in range(config["training"]["epochs"]):
        train_loss = train_one_epoch(model, loaders["train"], loss_fn, optimizer, device)
        val_loss, y_true, y_pred = evaluate(model, loaders["val"], loss_fn, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": accuracy(y_true, y_pred),
            "val_macro_f1": macro_f1(y_true, y_pred)
        }
        history.append(row)
        print(f"epoch {epoch}: train {train_loss:.4f} val {val_loss:.4f} "
              f"acc {row['val_accuracy']:.4f} f1 {row['val_macro_f1']:.4f}")

        if best_value is None or row["val_macro_f1"] > best_value:
            best_value, best_epoch = row["val_macro_f1"], epoch
            bad_epochs = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, run_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stopping at epoch {epoch}")
                break

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    return {"history": history, "best_epoch": best_epoch, "best_value": best_value, "device": device}

