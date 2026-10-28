"""Evidence for A1 Section 11, from recorded runs, code shapes, and saved plots.

No fitting or checkpoint inference. Tesseract reads existing plot annotations;
pixel matching identifies their original test images. Cached OCR/matches are
reused only when all source hashes agree.
Run: MPL_IGNORE_SYSTEM_FONTS=1 MPLCONFIGDIR=/tmp/a1-mpl-bundled .venv/bin/python scripts/a1_report/analyze_a1_comparison.py
"""
from __future__ import annotations

import difflib
import hashlib
import io
import json
import re
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from PIL import Image

from plot_a1_training_behaviour import RUNS
from plot_a1_qualitative import load_test
from summarize_a1_quantitative import CLASSES, read_runs

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/a1/assets/comparison"
TABLES = ROOT / "results/a1/tables"
UPPER = [0, 2, 3, 4, 6]  # Explicit class grouping, 50% of test support.
SHARED_IDS = [12, 7, 23, 40]


def source_examples(images, labels, raw_hashes):
    hashes = dict(raw_hashes)
    for _, run_id, _ in RUNS[:5]:
        for kind in ("correct", "wrong"):
            path = ROOT / f"results/a1/runs/{run_id}/figures/{kind}_test.png"
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    cache = TABLES / "a1_comparison_examples.json"
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved["source_sha256"] == hashes:
            return saved
    reference = images.astype(np.float32)
    lo = reference.min(axis=(1, 2), keepdims=True)
    reference = (reference - lo) / (reference.max(axis=(1, 2), keepdims=True) - lo) * 255
    normalize = lambda text: re.sub("[^a-z]", "", text.lower())

    def label_from_ocr(text, prefix):
        assert text.lower().startswith(prefix), text
        suffix = normalize(text[len(prefix):])
        scores = [difflib.SequenceMatcher(None, suffix, normalize(n)).ratio() for n in CLASSES]
        index = int(np.argmax(scores))
        assert scores[index] >= .85, (text, scores)
        return index, scores[index]

    panels, image_hashes = [], {}
    for model, run_id, _ in RUNS[:5]:
        for kind in ("correct", "wrong"):
            relative = f"results/a1/runs/{run_id}/figures/{kind}_test.png"
            grid = Image.open(ROOT / relative).convert("L")
            assert grid.size == (1200, 1200)
            if kind == "correct":
                image_hashes[model] = hashlib.sha256(grid.tobytes()).hexdigest()
            for panel in range(16):
                col, row = panel % 4, panel // 4
                x, y = [44, 338, 632, 926][col], [64, 359, 653, 948][row]
                tile = np.asarray(grid.crop((x, y, x + 230, y + 230)).resize(
                    (28, 28), Image.Resampling.NEAREST), dtype=np.float32)
                mse = ((reference - tile[None]) ** 2).mean(axis=(1, 2))
                closest = np.argsort(mse)[:2]
                idx = int(closest[0])
                assert mse[idx] < .25 and mse[closest[1]] > 100
                true, pred, ocr, score = int(labels[idx]), int(labels[idx]), "", 1.
                if kind == "wrong":
                    top = [0, 295, 589, 884][row]
                    title = grid.crop((294 * col, top, 294 * (col + 1), y)).resize(
                        (1176, 4 * (y - top)))
                    buffer = io.BytesIO()
                    title.save(buffer, format="PNG")
                    result = subprocess.run(
                        ["/opt/homebrew/bin/tesseract", "stdin", "stdout", "--psm", "6", "-l", "eng"],
                        input=buffer.getvalue(), capture_output=True, check=True)
                    ocr = result.stdout.decode().strip()
                    lines = ocr.splitlines()
                    assert len(lines) == 2, ocr
                    recovered_true, _ = label_from_ocr(lines[0], "true")
                    pred, score = label_from_ocr(lines[1], "pred")
                    assert recovered_true == true and pred != true
                panels.append(dict(model=model, run_id=run_id, kind=kind,
                                   source_grid=relative, source_panel=panel + 1,
                                   test_index=idx, true=true, pred=pred, ocr=ocr,
                                   ocr_label_similarity=score, match_mse=float(mse[idx]),
                                   runner_up_mse=float(mse[closest[1]])))
        print(f"Matched and read saved examples: {model}", flush=True)
    for model, _, _ in RUNS[:5]:
        subset = [p for p in panels if p["model"] == model]
        assert len({p["test_index"] for p in subset}) == 32
    saved = dict(purpose="Existing figure annotations and source-image identity only; no inference.",
                 source_sha256=hashes, correct_grid_pixel_sha256=image_hashes, panels=panels)
    cache.write_text(json.dumps(saved, indent=2) + "\n")
    return saved


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)


def plot_error_differences(runs):
    by_name = {r["model"]: r for r in runs}
    cnn_misses = 1000 - np.diag(by_name["CNN"]["confusion"])
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.7), sharey=True, layout="constrained")
    for ax, name in zip(axes, ["MLP", "LSTM rows", "Transformer"]):
        diff = 1000 - np.diag(by_name[name]["confusion"]) - cnn_misses
        ax.barh(range(10), diff, color=["#168878" if v >= 0 else "#CE7643" for v in diff], height=.65)
        for i, value in enumerate(diff):
            ax.text(value + (2 if value >= 0 else -2), i, f"{value:+d}",
                    ha="left" if value >= 0 else "right", va="center", fontsize=10)
        ax.axvline(0, color="#253649", linewidth=.8)
        ax.set(title=f"CNN vs {name}\n{int(diff.sum())} fewer errors in total",
               yticks=range(10), yticklabels=CLASSES, xlim=(-58, 132),
               xlabel="Other misses minus CNN misses")
        ax.set_xticks([-40, 0, 40, 80, 120])
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(length=0)
    axes[0].invert_yaxis()
    save(fig, "class_error_differences")


def front(runs, cost):
    return [r["model"] for r in runs if not any(
        s[cost] <= r[cost] and s["macro_f1"] >= r["macro_f1"] and
        (s[cost] < r[cost] or s["macro_f1"] > r["macro_f1"])
        for s in runs)]


def plot_efficiency(runs):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9), sharey=True, layout="constrained")
    labels = {"LSTM rows": "Rows", "LSTM columns": "Cols", "LSTM patches": "Patches"}
    # Offsets are display choices only; data positions are unchanged.
    offsets = {
        "params": [(6, 4), (6, 5), (7, 3), (7, 7), (-80, 10), (-5, 8), (8, -18)],
        "inference_ms_per_image": [(6, 4), (-26, -19), (-12, 9), (8, 4), (-76, -17), (8, -2), (-37, -23)],
    }
    for ax, cost in zip(axes, ["params", "inference_ms_per_image"]):
        scale = 1e-3 if cost == "params" else 1e3
        selected = front(runs, cost)
        points = sorted([r for r in runs if r["model"] in selected], key=lambda r: r[cost])
        ax.plot([r[cost] * scale for r in points], [100*r["macro_f1"] for r in points],
                linestyle="--", color="#A3AFBA", linewidth=1, zorder=1)
        for i, run in enumerate(runs):
            color = RUNS[i][2]
            x, y = run[cost] * scale, 100 * run["macro_f1"]
            ax.scatter(x, y, color=color, s=70, edgecolor="white", linewidth=.8, zorder=4,
                       marker="D" if run["model"] in selected else "o")
            ax.annotate(labels.get(run["model"], run["model"]), (x, y),
                         xytext=offsets[cost][i], textcoords="offset points", fontsize=10,
                         arrowprops=dict(arrowstyle="-", color=color, linewidth=.6))
        ax.set_xscale("log")
        ax.set_ylim(83, 93)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.grid(alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set(title="(a) Parameters and test macro-F1", xlabel="Parameters (thousands, log scale)",
                ylabel="Test macro-F1", xlim=(5.5, 900), xticks=[8, 50, 100, 500])
    axes[1].set(title="(b) Inference and test macro-F1", xlabel="Batched time per image (microseconds, log scale)",
                xlim=(.35, 21), xticks=[.5, 1, 2, 5, 10, 20])
    for ax in axes:
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
    save(fig, "quality_cost_frontiers")


def plot_shared(images, examples):
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 3.0))
    fig.subplots_adjust(left=.02, right=.98, top=.88, bottom=.19, wspace=.18)
    for ax, idx in zip(axes, SHARED_IDS):
        rows = [p for p in examples["panels"] if p["test_index"] == idx]
        assert len(rows) == 5 and len({r["model"] for r in rows}) == 5
        true = rows[0]["true"]
        ax.imshow(images[idx], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.axis("off")
        ax.set_title(f"Test #{idx}", fontsize=12, weight="bold")
        ax.text(.5, -.08, f"True: {CLASSES[true]}", fontsize=11.5, ha="center", va="top",
                 transform=ax.transAxes)
    save(fig, "shared_test_examples")
    lines = ["% Predictions from identified saved example panels, not new inference.",
             r"\begin{tabularx}{\textwidth}{@{}lYYYY@{}}", r"\toprule",
             r"Configuration & Test \#12 & Test \#7 & Test \#23 & Test \#40 \\", r"\midrule"]
    for model, _, _ in RUNS[:5]:
        cells = [model]
        for idx in SHARED_IDS:
            row, = [p for p in examples["panels"] if p["model"] == model and p["test_index"] == idx]
            label = CLASSES[row["pred"]]
            cells.append(r"\textbf{"+label+"}" if row["pred"] == row["true"] else label)
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    (OUT / "shared_predictions.tex").write_text("\n".join(lines) + "\n")


def arithmetic():
    # Multiply-accumulate counts for learned affine and attention matrix products.
    # Excludes biases, activations, normalization, pooling, softmax, data movement.
    n, d, m = 17, 128, 256
    block = dict(qkv_and_output=4*n*d*d, attention_products=2*n*n*d, feed_forward=2*n*d*m)
    macs = {
        "Linear": 784*10,
        "MLP": 784*256 + 256*128 + 128*10,
        "CNN": 28*28*32*9 + 14*14*64*32*9 + 3136*10,
        "LSTM rows": 28*4*128*(28+128) + 128*10,
        "LSTM columns": 28*4*128*(28+128) + 128*10,
        "LSTM patches": 16*4*128*(49+128) + 128*10,
        "Transformer": 16*49*128 + 4*sum(block.values()) + 128*10,
    }
    lines = ["% Structural arithmetic estimates, not runtime profiling.",
             r"\begin{tabularx}{\textwidth}{@{}Yr@{}}", r"\toprule",
             r"Configuration & MACs per image \\", r"\midrule"]
    for name, _, _ in RUNS:
        lines.append(name+f" & {macs[name]:,}"+r" \\")
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    (OUT / "arithmetic_cost.tex").write_text("\n".join(lines)+"\n")
    return dict(macs=macs,
                lstm_matrix_products=dict(input_to_gates_all_representations=784*4*128,
                                          hidden_to_gates_rows_or_columns=28*4*128*128,
                                          hidden_to_gates_patches=16*4*128*128,
                                          classifier=128*10),
                transformer_per_block=block,
                attention_products_fraction=block["attention_products"]/sum(block.values()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 12, "axes.labelsize": 10,
                         "text.color": "#253649", "axes.titleweight": "bold"})
    runs = read_runs()
    images, labels, hashes = load_test()
    examples = source_examples(images, labels, hashes)
    plot_error_differences(runs)
    plot_efficiency(runs)
    plot_shared(images, examples)
    costs = arithmetic()
    by_name = {r["model"]: r for r in runs}
    pairs = []
    for better, base in [("MLP", "Linear"), ("CNN", "Linear"), ("CNN", "MLP"),
                         ("CNN", "LSTM rows"), ("CNN", "Transformer"),
                         ("LSTM rows", "LSTM columns"), ("LSTM rows", "LSTM patches")]:
        x, y = by_name[better], by_name[base]
        net = np.diag(x["confusion"]) - np.diag(y["confusion"])
        assert sum(net) == y["errors"] - x["errors"]
        pairs.append(dict(model=better, baseline=base,
                          accuracy_gain_pp=100*(x["accuracy"]-y["accuracy"]),
                          macro_f1_gain_pp=100*(x["macro_f1"]-y["macro_f1"]),
                          net_fewer_errors=int(sum(net)),
                          relative_error_reduction=float(sum(net)/y["errors"]),
                          net_error_reduction_by_true_class=net.tolist(),
                          upper_true_class_net_reduction=int(sum(net[k] for k in UPPER))))
    groups = []
    for run in runs:
        cm = run["confusion"]
        within = sum(cm[i][j] for i in UPPER for j in UPPER if i != j)
        groups.append(dict(model=run["model"], total_errors=run["errors"],
                           within_upper_group_errors=within,
                           within_upper_group_fraction=within/run["errors"],
                           errors_from_true_upper_classes=sum(1000-cm[k][k] for k in UPPER)))
    histories = []
    for run in runs:
        train = json.loads((ROOT/f"results/a1/runs/{run['run_id']}/metrics.json").read_text())
        histories.append(dict(model=run["model"], mean_fit_seconds_per_completed_epoch=
                              train["train_time"]/train["epochs_run"]))
    facts = dict(
        purpose="Analytical comparisons of recorded runs and structural operation counts; no new training or inference.",
        upper_group=dict(class_ids=UPPER, names=[CLASSES[k] for k in UPPER], test_support=5000),
        pairwise=pairs, error_groups=groups, arithmetic=costs,
        frontiers=dict(parameters_macro_f1=front(runs, "params"),
                       inference_macro_f1=front(runs, "inference_ms_per_image")),
        fit_time_per_epoch=histories, shared_test_ids=SHARED_IDS,
        evidence_limits="Net count changes do not identify corrected individual images. Shared examples are selected panels, not full paired predictions or a significance test. MAC counts are structural estimates, not measured FLOPs or kernel times.")
    (TABLES / "a1_comparison_analysis.json").write_text(json.dumps(facts, indent=2)+"\n")
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
