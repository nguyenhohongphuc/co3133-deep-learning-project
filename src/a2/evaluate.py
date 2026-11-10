"""Evaluate an A2 checkpoint and export one prediction row per image.

Run from the repository root:
    python -m src.a2.evaluate --run-dir results/a2/runs/<run_id> --split val

"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import time

from PIL import Image
import torch
from torch.utils.data import DataLoader
import yaml

from src.a2.data import VNFoods, build_transforms, load_split
from src.a2.models import MODELS, count_params
from src.utils.env import get_env_info
from src.utils.git_commit import git_commit
from src.utils.seed import set_seed, worker_init_fn


REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTION_FIELDS = [
    "run_id", "model", "seed", "split", "index", "path", "md5",
    "true_label", "true_class", "pred_label", "pred_class", "correct",
    "confidence", "true_class_probability",
    "top2_label", "top2_class", "top2_probability",
    "top3_label", "top3_class", "top3_probability", "confidence_margin",
    "width", "height", "aspect_ratio",
]


class PredictionDataset(VNFoods):
    """Keep split indices and original image dimensions alongside model input."""

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        with Image.open(self.root / self.items[index]["path"]) as original:
            width, height = original.size
        return image, label, index, width, height


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", "--run_dir", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--checkpoint", type=Path, help="Default: <run-dir>/best.pt")
    parser.add_argument("--data-dir", type=Path, help="Override the saved image root")
    parser.add_argument("--split-file", type=Path, help="Override the saved split JSON")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: <run-dir>/evaluation/<split>")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto",
                        help="Evaluation device, independent of the training device")
    parser.add_argument("--batch-size", type=int, help="Override saved batch size")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (default: 0, portable across OSes)")
    args = parser.parse_args(argv)
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    return args


def _saved_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu or auto")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; use --device cpu or auto")
    return torch.device(name)


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _class_metrics(confusion, classes):
    rows = []
    for label, name in enumerate(classes):
        tp = int(confusion[label, label])
        support = int(confusion[label].sum())
        predicted = int(confusion[:, label].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        rows.append(dict(label=label, **{"class": name}, support=support,
                         predicted=predicted, correct=tp, precision=precision,
                         recall=recall, f1=f1))
    return rows


def evaluate_run(args):
    run_dir = args.run_dir.expanduser().resolve()
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Saved run config not found: {config_path}")
    checkpoint = (args.checkpoint.expanduser().resolve() if args.checkpoint
                  else run_dir / "best.pt")
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. Add best.pt to the run folder "
            "or pass --checkpoint /path/to/best.pt.")

    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    metrics_path = run_dir / "metrics.json"
    training_metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    model_name = config.get("model", training_metrics.get("model"))
    if model_name not in MODELS:
        raise ValueError(f"Unknown A2 model in saved run: {model_name!r}")
    if training_metrics.get("model", model_name) != model_name:
        raise ValueError("Model differs between config.yaml and metrics.json")
    model_config = config["models"][model_name]
    # The saved top-level training block is already resolved by train.py.
    # Reapplying models.<name>.training would undo historical CLI overrides.
    batch_size = args.batch_size if args.batch_size is not None else config["training"]["batch_size"]
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Saved batch_size must be a positive integer")
    seed = config["seed"]
    set_seed(seed)
    device = _device(args.device)
    data_dir = args.data_dir.expanduser().resolve() if args.data_dir else _saved_path(config["data_dir"])
    split_path = args.split_file.expanduser().resolve() if args.split_file else _saved_path(config["split_file"])
    split = load_split(split_path)
    classes, items = split["classes"], split[args.split]
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("Split must contain at least two unique class names")
    if not items:
        raise ValueError(f"Selected split {args.split!r} is empty")
    for item in items:
        if not isinstance(item["label"], int) or not 0 <= item["label"] < len(classes):
            raise ValueError(f"Invalid label in split: {item}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {data_dir}; use --data-dir")

    params = dict(model_config.get("params", {}))
    if model_name.startswith("resnet50_"):
        params["pretrained"] = False  # best.pt supplies every parameter; stay offline.
    model = MODELS[model_name](**params)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    dataset = PredictionDataset(data_dir, items,
                                build_transforms(config, model_config["normalize"], train=False))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda",
                        worker_init_fn=worker_init_fn,
                        generator=torch.Generator().manual_seed(seed))
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir
                  else run_dir / "evaluation" / args.split)
    if output_dir == run_dir:
        raise ValueError("Use an evaluation subfolder for --output-dir to preserve training metrics.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    confusion = torch.zeros(len(classes), len(classes), dtype=torch.int64)
    total_loss = forward_seconds = 0.0
    seen = top3_correct = 0
    run_id = training_metrics.get("run_id", run_dir.name)
    start = time.perf_counter()

    # Publish files only after inference and metric computation succeed.
    with tempfile.TemporaryDirectory(prefix=".a2-eval-", dir=output_dir) as temporary:
        temporary = Path(temporary)
        with (temporary / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=PREDICTION_FIELDS)
            writer.writeheader()
            with torch.inference_mode():
                for images, labels, indices, widths, heights in loader:
                    images = images.to(device, non_blocking=device.type == "cuda")
                    if seen == 0:
                        model(images)  # One untimed warmup batch.
                    _synchronize(device)
                    tick = time.perf_counter()
                    logits = model(images)
                    _synchronize(device)
                    forward_seconds += time.perf_counter() - tick
                    if tuple(logits.shape) != (len(labels), len(classes)):
                        raise ValueError(f"Model output {tuple(logits.shape)} does not match "
                                         f"batch/classes {(len(labels), len(classes))}")
                    logits = logits.float().cpu()
                    if not torch.isfinite(logits).all():
                        raise ValueError("Checkpoint produced non-finite logits")
                    total_loss += torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
                    probabilities = logits.softmax(dim=1)
                    scores, predictions = probabilities.topk(min(3, len(classes)), dim=1)
                    predicted = predictions[:, 0]
                    confusion += torch.bincount(labels * len(classes) + predicted,
                                                minlength=len(classes) ** 2).reshape(len(classes), -1)
                    top3_correct += int((predictions == labels[:, None]).any(dim=1).sum())
                    for row_index, index in enumerate(indices.tolist()):
                        label, pred = int(labels[row_index]), int(predicted[row_index])
                        width, height = int(widths[row_index]), int(heights[row_index])
                        row = dict(run_id=run_id, model=model_name, seed=seed, split=args.split,
                                   index=index, path=items[index]["path"], md5=items[index].get("md5", ""),
                                   true_label=label, true_class=classes[label], pred_label=pred,
                                   pred_class=classes[pred], correct=int(label == pred),
                                   confidence=float(scores[row_index, 0]),
                                   true_class_probability=float(probabilities[row_index, label]),
                                   confidence_margin=float(scores[row_index, 0] - scores[row_index, 1]),
                                   width=width, height=height, aspect_ratio=width / height)
                        for rank in (2, 3):
                            if rank <= predictions.shape[1]:
                                alternative = int(predictions[row_index, rank - 1])
                                row.update({f"top{rank}_label": alternative,
                                            f"top{rank}_class": classes[alternative],
                                            f"top{rank}_probability": float(scores[row_index, rank - 1])})
                        writer.writerow(row)
                    seen += len(labels)
        if seen != len(items):
            raise ValueError(f"Expected {len(items)} predictions, got {seen}")
        per_class = _class_metrics(confusion, classes)
        total, trainable = count_params(model)
        results = dict(
            run_id=run_id, model=model_name, seed=seed, split=args.split, num_samples=seen,
            classes=classes, accuracy=int(confusion.diag().sum()) / seen,
            macro_f1=sum(row["f1"] for row in per_class) / len(classes),
            weighted_f1=sum(row["f1"] * row["support"] for row in per_class) / seen,
            top3_accuracy=top3_correct / seen, loss=total_loss / seen,
            confusion=confusion.tolist(), device=str(device), batch_size=batch_size,
            num_workers=args.num_workers, param=total, param_trainable=trainable,
            inference_forward_seconds=forward_seconds,
            inference_ms_per_image=forward_seconds / seen * 1000,
            inference_images_per_second=seen / forward_seconds,
            inference_timing="One pass, model forward only, one untimed warmup batch; "
                             "excludes loading, device transfers, softmax and CSV export",
            evaluation_seconds=time.perf_counter() - start,
            training_commit=training_metrics.get("commit"), evaluation_commit=git_commit(),
            env=get_env_info(device), run_dir=str(run_dir),
            checkpoint=str(checkpoint), checkpoint_sha256=_sha256(checkpoint),
            config_file=str(config_path), config_sha256=_sha256(config_path),
            split_file=str(split_path), split_sha256=_sha256(split_path), data_dir=str(data_dir),
            normalization=model_config["normalize"], evaluation_data_config=config["data"],
            macro_f1_policy="All classes in split.classes, zero F1 for undefined classes",
        )
        (temporary / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n",
                                                encoding="utf-8")
        with (temporary / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(per_class[0]))
            writer.writeheader()
            writer.writerows(per_class)
        with (temporary / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["true_class / predicted_class", *classes])
            writer.writerows([name, *confusion[i].tolist()] for i, name in enumerate(classes))
        for filename in ("predictions.csv", "metrics.json", "per_class_metrics.csv", "confusion_matrix.csv"):
            (temporary / filename).replace(output_dir / filename)
    return results, output_dir


def main(argv=None):
    args = parse_args(argv)
    try:
        results, output_dir = evaluate_run(args)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        raise SystemExit(f"A2 evaluation failed: {error}") from error
    print(json.dumps({key: results[key] for key in
                      ("run_id", "split", "num_samples", "accuracy", "macro_f1")}, ensure_ascii=False))
    print(f"Saved evaluation to {output_dir}")


if __name__ == "__main__":
    main()
