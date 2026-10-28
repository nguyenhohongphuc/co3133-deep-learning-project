"""Trace saved CNN example grids to original test images and replot selected cases.

Predictions are transcribed from annotations in the existing correct/wrong grids,
not produced by a model. Pixel matching recovers test IDs, not predictions.
Run: MPL_IGNORE_SYSTEM_FONTS=1 MPLCONFIGDIR=/tmp/a1-mpl-bundled .venv/bin/python scripts/a1_report/plot_a1_qualitative.py
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/a1/assets/qualitative"
RUN_ID = "a1_cnn_s42_1027-0157"
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
# Read visually from the saved 4 x 4 grids, in row-major order.
ANNOTATIONS = {
    "correct": {
        "true": [9, 2, 1, 1, 6, 1, 4, 6, 5, 7, 4, 5, 7, 3, 4, 1],
        "pred": [9, 2, 1, 1, 6, 1, 4, 6, 5, 7, 4, 5, 7, 3, 4, 1],
    },
    "wrong": {
        "true": [9, 4, 3, 6, 3, 2, 3, 9, 4, 9, 6, 4, 6, 2, 3, 9],
        "pred": [5, 6, 1, 0, 6, 6, 4, 7, 2, 7, 3, 2, 2, 6, 6, 7],
    },
}
# Zero-based source-grid panel indices, deliberately chosen for visual discussion.
SELECTIONS = {
    "correct_examples": ("correct", [0, 2, 4, 8, 9, 13]),
    "difficult_correct": ("correct", [1, 6, 7, 10]),
    "incorrect_examples": ("wrong", [0, 1, 2, 3, 7, 8, 10, 13]),
}


def load_test():
    image_path = ROOT / "data/FashionMNIST/raw/t10k-images-idx3-ubyte"
    label_path = ROOT / "data/FashionMNIST/raw/t10k-labels-idx1-ubyte"
    image_bytes, label_bytes = image_path.read_bytes(), label_path.read_bytes()
    assert struct.unpack(">IIII", image_bytes[:16]) == (2051, 10000, 28, 28)
    assert struct.unpack(">II", label_bytes[:8]) == (2049, 10000)
    images = np.frombuffer(image_bytes[16:], np.uint8).reshape(10000, 28, 28)
    labels = np.frombuffer(label_bytes[8:], np.uint8)
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(b).hexdigest()
              for p, b in [(image_path, image_bytes), (label_path, label_bytes)]}
    return images, labels, hashes


def trace_grids(images, labels):
    # The original imshow call auto-scales each image. Reproduce that display
    # scaling only for identification; the final figures use raw 0..255 values.
    reference = images.astype(np.float32)
    lo, hi = reference.min(axis=(1, 2), keepdims=True), reference.max(axis=(1, 2), keepdims=True)
    reference = (reference - lo) / (hi - lo) * 255
    sources, records = {}, {}
    xs, ys = [44, 338, 632, 926], [64, 359, 653, 948]
    for kind, annotations in ANNOTATIONS.items():
        path = ROOT / f"results/a1/runs/{RUN_ID}/figures/{kind}_test.png"
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        grid = Image.open(path).convert("L")
        assert grid.size == (1200, 1200)
        rows = []
        for panel in range(16):
            x, y = xs[panel % 4], ys[panel // 4]
            tile = np.asarray(grid.crop((x, y, x + 230, y + 230)).resize(
                (28, 28), Image.Resampling.NEAREST), dtype=np.float32)
            mse = np.mean((reference - tile[None]) ** 2, axis=(1, 2))
            matches = np.argsort(mse)[:2]
            idx = int(matches[0])
            assert mse[idx] < 0.25 and mse[matches[1]] > 100
            assert int(labels[idx]) == annotations["true"][panel]
            pred = annotations["pred"][panel]
            assert (pred == int(labels[idx])) == (kind == "correct")
            rows.append(dict(test_index=idx, true=int(labels[idx]), pred=pred,
                             true_name=CLASSES[int(labels[idx])], pred_name=CLASSES[pred],
                             source_grid=str(path.relative_to(ROOT)),
                             source_row=panel // 4 + 1, source_column=panel % 4 + 1,
                             match_mse=float(mse[idx]), runner_up_mse=float(mse[matches[1]])))
        assert [r["test_index"] for r in rows] == sorted(r["test_index"] for r in rows)
        assert len({r["test_index"] for r in rows}) == 16
        records[kind] = rows
    assert not ({r["test_index"] for r in records["correct"]} &
                {r["test_index"] for r in records["wrong"]})
    return records, sources


def plot_examples(images, selected, name):
    nrows = 1 if len(selected) == 4 else 2
    ncols = 3 if len(selected) == 6 else 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, (3.05 if ncols == 3 else 3.2) * nrows), squeeze=False)
    fig.subplots_adjust(left=.025, right=.975, top=.90 if nrows == 1 else .95,
                        bottom=.20 if nrows == 1 else .11, wspace=.18,
                        hspace=.60 if ncols == 3 else .67)
    for letter, ax, case in zip("abcdefgh", axes.flat, selected):
        ax.imshow(images[case["test_index"]], cmap="gray", vmin=0, vmax=255,
                  interpolation="nearest")
        ax.set_title(f"({letter}) Test #{case['test_index']}", fontsize=12,
                     color="#253649", weight="bold", pad=8)
        ax.axis("off")
        ax.text(.5, -.055, f"True: {case['true_name']}", ha="center", va="top",
                 fontsize=11.5, color="#253649", transform=ax.transAxes)
        ax.text(.5, -.17, f"Pred: {case['pred_name']}", ha="center", va="top",
                 fontsize=11.5, color="#168878" if case["true"] == case["pred"] else "#B04D32",
                 transform=ax.transAxes)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    images, labels, hashes = load_test()
    records, sources = trace_grids(images, labels)
    # A second annotation audit is retained by the cross-model comparison.
    comparison_path = ROOT / "results/a1/tables/a1_comparison_examples.json"
    crosschecked = False
    if comparison_path.exists():
        comparison = json.loads(comparison_path.read_text())
        for rows in records.values():
            for row in rows:
                source, = [p for p in comparison["panels"]
                           if p["model"] == "CNN" and p["test_index"] == row["test_index"]]
                assert (row["true"], row["pred"]) == (source["true"], source["pred"])
        crosschecked = True
    selected = {}
    for name, (kind, indices) in SELECTIONS.items():
        selected[name] = [dict(panel=letter, **records[kind][idx])
                          for letter, idx in zip("abcdefgh", indices)]
        plot_examples(images, selected[name], name)
    assert not ({r["test_index"] for r in selected["correct_examples"]} &
                {r["test_index"] for r in selected["difficult_correct"]})
    audit = dict(
        run_id=RUN_ID, model="CNN", seed=42,
        purpose="Trace saved examples to original test pixels and replot; no new inference.",
        prediction_provenance="Transcribed from saved grid annotations; checked against the cross-model OCR annotation audit when present. Test #32 is Dress predicted as Trouser.",
        crosschecked_against_annotation_audit=crosschecked,
        matching="Grid tiles downsampled to 28x28 and compared with per-image auto-scaled test images; best MSE < 0.25 and runner-up MSE > 100 on 0..255 intensities.",
        display="Original test pixels with fixed grayscale 0..255 and nearest-neighbor display; no enhancement.",
        selection="Six correct and eight incorrect cases chosen from the saved first-16 grids; four further visually challenging correct cases are drawn from the same correct grid, with no overlap with the six ordinary examples and no difficulty ranking.",
        limitations="No confidence scores, full prediction vector, saliency, or new checkpoint evaluation used.",
        source_sha256={**hashes, **sources}, matched_source_panels=records, figures=selected)
    target = ROOT / "results/a1/tables/a1_qualitative_examples.json"
    target.write_text(json.dumps(audit, indent=2) + "\n")
    print("Matched all 32 saved panels uniquely to original test images; no inference.")
    for name, cases in selected.items():
        print(name, [(r["panel"], r["test_index"], r["true_name"], r["pred_name"]) for r in cases])


if __name__ == "__main__":
    main()
