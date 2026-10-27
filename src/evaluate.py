
from __future__ import annotations

import argparse
import json
import pathlib
from sklearn.metrics import confusion_matrix
import yaml
import time

import numpy as np
import torch

from src.data.loaders import build_loaders
from src.models import MODELS
from src.utils.seed import set_seed
from src.engine.trainer import resolve_device, build_loss, accuracy, macro_f1, evaluate
from src.engine.plots import plot_curves, plot_confusion, plot_examples

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    return args


def measure_inference(model, loader, device, batch_size=128, warmup=3, repeats=5):
    """Forward-only time over the whole test set (handbook 12.1).

    Data is moved to the device once beforehand, so loading, normalization and
    host-to-device copies are not timed. Returns the median over `repeats` passes.
    """
    images = torch.cat([x for x, _ in loader]).to(device)
    batches = images.split(batch_size)

    def sync():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            model(batches[0])
        sync()
        for _ in range(repeats):
            start = time.perf_counter()
            for batch in batches:
                model(batch)
            sync()
            times.append(time.perf_counter() - start)

    total = float(np.median(times))
    return {
        "inference_time": total,
        "inference_ms_per_image": total / len(images) * 1000,
        "inference_batch_size": batch_size,
        "inference_repeats": repeats,
    }



def main():
    args = parse_args()
    run_dir = args.run_dir
    with open(run_dir / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(args.run_dir / "metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    set_seed(metrics["seed"])
    device = resolve_device(config)
    model = MODELS[metrics["model"]](**config["models"][metrics["model"]])
    state = torch.load(run_dir/"best.pt",map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    
    train_loader, val_loader, test_loader = build_loaders(config)
    loss_fn = build_loss(config)
    _, y_true, y_pred = evaluate(model, test_loader, loss_fn, device)
    results = {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "confusion": confusion_matrix(y_true, y_pred, labels=range(10)).tolist(),
        **measure_inference(model, test_loader, device,
                            batch_size=config["training"]["batch_size"]),
        "device": str(device),
        "model": metrics["model"],
    } 
    with open(run_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    plot_curves(run_dir / "history.csv", fig_dir / "curves.png")
    plot_confusion(results["confusion"], CLASS_NAMES, fig_dir / "confusion_test.png")
    plot_examples(test_loader.dataset, y_true, y_pred,
                  config["data"]["mean"][0], config["data"]["std"][0],
                  CLASS_NAMES, fig_dir, "test")
    
if __name__ == "__main__":
    main()
