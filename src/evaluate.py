"""Evaluate a checkpoint and write every required output. Owner: TV 2.

    python -m src.evaluate --config configs/a1/linear.yaml \
        --checkpoint results/a1/runs/a1_linear_s42_0923-1900/best.pt

Produces the handbook 12.1 outputs: accuracy, macro-F1, parameter count, training and
inference time, confusion matrix, correct / incorrect examples.

Writes metrics_<split>.json next to the checkpoint and adds the headline numbers back
into that run's metrics.json, so scripts/make_tables.py reads one file per run.

Run it on --split test ONCE per model, at the end. Every test-set look that changes a
decision turns the test set into a second validation set.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch
from torchvision.datasets import FashionMNIST

import src.models  # noqa: F401  - importing the package registers every model
from src.common.io import save_json
from src.common.registry import build_model
from src.common.seed import set_seed
from src.data.loaders import build_loaders
from src.engine.metrics import summarize
from src.engine.plots import plot_confusion, plot_curves, plot_predictions
from src.engine.profiler import count_parameters, inference_time
from src.engine.trainer import build_loss, evaluate, resolve_device
from src.train import load_config


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    return p.parse_args()


def unnormalize(images, config):
    """Undo transforms.Normalize so the example grid is readable instead of grey."""
    mean = config["data"]["mean"][0]
    std = config["data"]["std"][0]
    return (images * std + mean).clamp(0, 1).numpy()


def example_grid(loader, indices, config):
    """Fetch just the images we plot, straight from the dataset behind the loader."""
    images = torch.stack([loader.dataset[int(i)][0] for i in indices])
    return unnormalize(images, config)


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])

    device = resolve_device(config)
    train_loader, val_loader, test_loader = build_loaders(config)
    loader = test_loader if args.split == "test" else val_loader

    model = build_model(config)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model_state"])

    loss_fn = build_loss(config)
    loss, y_true, y_pred = evaluate(model, loader, loss_fn, device)  # reuse the trainer's

    results = summarize(y_true, y_pred, num_classes=10)
    results["split"] = args.split
    results["loss"] = loss
    results["params"] = count_parameters(model)
    results["inference"] = inference_time(model, loader, device)

    run_folder = pathlib.Path(args.checkpoint).parent
    run_name = config["run_name"]
    save_json(run_folder / f"metrics_{args.split}.json", results)

    # make_tables.py reads one metrics.json per run, so fold the headline numbers in.
    summary_path = run_folder / "metrics.json"
    if summary_path.exists() and args.split == "test":
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["accuracy"] = results["accuracy"]
        summary["macro_f1"] = results["macro_f1"]
        summary["inference_ms_per_image"] = results["inference"]["ms_per_image"]
        save_json(summary_path, summary)

    class_names = FashionMNIST(root=config["data_dir"], train=False).classes
    figures = pathlib.Path(f"results/{run_name.split('_')[0]}/figures")
    figures.mkdir(parents=True, exist_ok=True)

    plot_confusion(results["confusion"], class_names,
                   figures / f"{run_name}_confusion_{args.split}.png")

    history_csv = run_folder / "history.csv"
    if history_csv.exists():
        plot_curves(history_csv, figures / f"{run_name}_curves.png")

    # Pick the examples first, then plot them: the failures are spread over the split.
    correct = np.flatnonzero(y_true == y_pred)[:16]
    wrong = np.flatnonzero(y_true != y_pred)[:16]
    plot_predictions(example_grid(loader, correct, config), y_true[correct], y_pred[correct],
                     class_names, figures / f"{run_name}_correct_{args.split}.png")
    if len(wrong):
        plot_predictions(example_grid(loader, wrong, config), y_true[wrong], y_pred[wrong],
                         class_names, figures / f"{run_name}_wrong_{args.split}.png")

    print(f"{run_name} on {args.split}: "
          f"accuracy {results['accuracy']:.4f}  macro-F1 {results['macro_f1']:.4f}  "
          f"loss {loss:.4f}")
    print(f"params {results['params']:,}  "
          f"inference {results['inference']['ms_per_image']:.3f} ms/image on {device}")
    print(f"figures -> {figures}")


if __name__ == "__main__":
    main()
