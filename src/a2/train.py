"""Train một model A2. Dùng lại src/engine/trainer.fit của A1 (không sửa).

    python -m src.a2.train --model cnn
    python -m src.a2.train --model resnet50_frozen --epochs 10
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import yaml

from src.a2.data import build_loaders
from src.a2.models import MODELS, count_params
from src.engine.plots import plot_curves
from src.engine.trainer import fit
from src.utils.env import get_env_info
from src.utils.git_commit import git_commit
from src.utils.run_dir import make_run_dir
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cnn", choices=list(MODELS))
    parser.add_argument("--config", type=str, default="configs/a2.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="ghi đè số epoch (vd. chạy thử)")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def resolve_config(config: dict, model_name: str, epochs: int | None, seed: int | None) -> dict:
    """Gộp models.<tên>.training vào training để trainer A1 đọc như cũ."""
    config = copy.deepcopy(config)
    model_cfg = config["models"][model_name]
    config["training"].update(model_cfg.get("training", {}))
    if epochs is not None:
        config["training"]["epochs"] = epochs
    if seed is not None:
        config["seed"] = seed
    config["model"] = model_name
    return config


def main():
    args = parse_args()
    commit = git_commit()
    with open(args.config, "r", encoding="utf-8") as f:
        config = resolve_config(yaml.safe_load(f), args.model, args.epochs, args.seed)
    set_seed(config["seed"])
    model_cfg = config["models"][args.model]

    run_id, run_dir = make_run_dir("a2", args.model, config["seed"])
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    model = MODELS[args.model](**model_cfg.get("params", {}))
    total, trainable = count_params(model)
    print(f"{run_id}: {total:,} params ({trainable:,} trainable)")

    train_loader, val_loader, _ = build_loaders(config, model_cfg["normalize"])
    start_time = time.time()
    result = fit(model, {"train": train_loader, "val": val_loader}, config, run_dir)
    train_time = time.time() - start_time

    best = result["history"][result["best_epoch"]]
    metrics = {
        "run_id": run_id,
        "model": args.model,
        "seed": config["seed"],
        "commit": commit,
        "device": result["device"],
        "env": get_env_info(result["device"]),
        "param": total,
        "param_trainable": trainable,
        "epochs_planned": config["training"]["epochs"],
        "epochs_run": len(result["history"]),
        "train_time": train_time,
        "best_epoch": result["best_epoch"],
        "best_value": result["best_value"],
        "monitor": config["training"]["monitor"],
        "val_accuracy": best["val_accuracy"],
        "val_macro_f1": best["val_macro_f1"],
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    plot_curves(run_dir / "history.csv", fig_dir / "curves.png")
    print(json.dumps({k: metrics[k] for k in ["run_id", "val_accuracy", "val_macro_f1", "best_epoch", "train_time"]}))


if __name__ == "__main__":
    main()
