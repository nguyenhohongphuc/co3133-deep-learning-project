"""Evaluate a finalized fusion checkpoint, without downloading encoder weights."""
from __future__ import annotations

import argparse

from transformers import AutoTokenizer

from src.a3.fusion_data import load_fusion_splits, repo_path
from src.a3.fusion_engine import (choose_device, load_checkpoint, make_loader, run_epoch,
                                  seed_all, write_csv, write_json)


def evaluate(checkpoint, *, split="validation", device_name="auto", image_root=None, data_dir=None):
    checkpoint = repo_path(checkpoint)
    model, ck = load_checkpoint(checkpoint, data_dir=data_dir)
    cfg = ck["config"]
    if image_root is not None:
        cfg["data"]["image_root"] = image_root
    if data_dir is not None:
        cfg["data"]["dir"] = data_dir
    seed_all(cfg["seed"])
    splits, labels = load_fusion_splits(cfg)
    names = [k for k, v in sorted(labels.items(), key=lambda item: item[1])]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint.parent / "tokenizer", local_files_only=True)
    device = choose_device(device_name)
    model.to(device)
    loader = make_loader(cfg, splits[split], tokenizer, device)
    metrics, predictions = run_epoch(model, loader, device, amp=cfg["train"]["amp"])
    metrics.update(checkpoint_epoch=ck["epoch"], classes=names)
    for row in metrics["per_class"]:
        row["class_name"] = names[row["label"]]
    for row in predictions:
        row["product_type"] = names[row["label"]]
        row["predicted_type"] = names[row["prediction"]]
        row["correct"] = row["label"] == row["prediction"]
    # Separate filenames for best.pt / last.pt to avoid overwriting another evaluation.
    out = checkpoint.parent / f"evaluation_{checkpoint.stem}_{split}"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metrics.json", metrics)
    write_csv(out / "predictions.csv", predictions)
    write_csv(out / "per_class.csv", metrics["per_class"])
    write_csv(out / "confusion_matrix.csv", [
        {"true_class": name, **dict(zip(names, values))}
        for name, values in zip(names, metrics["confusion_matrix"])])
    print(f"{split}: accuracy={metrics['accuracy']:.4f}, macro-F1={metrics['macro_f1']:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-root", help="Override root for manifest image_path on this host")
    parser.add_argument("--data-dir", help="Relocated copies of the identical CSV/label files")
    args = parser.parse_args()
    evaluate(args.checkpoint, split=args.split, device_name=args.device,
             image_root=args.image_root, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
