"""Replot saved A1 histories for report Section 8; no training or inference.

Run from the repository with a Python environment containing matplotlib:
    MPL_IGNORE_SYSTEM_FONTS=1 MPLCONFIGDIR=/tmp/a1-mpl .venv/bin/python scripts/a1_report/plot_a1_training_behaviour.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/a1/assets/training"
RUNS = [
    ("Linear", "a1_linear_s42_1027-0134", "#687587"),
    ("MLP", "a1_mlp_s42_1027-0143", "#B07822"),
    ("CNN", "a1_cnn_s42_1027-0157", "#168878"),
    ("LSTM rows", "a1_rnn_rows_s42_1027-0204", "#6454A4"),
    ("Transformer", "a1_transformer_patches_s42_1027-0242", "#2775B6"),
    ("LSTM columns", "a1_rnn_cols_s42_1027-0217", "#CE7643"),
    ("LSTM patches", "a1_rnn_patches_s42_1027-0229", "#35869A"),
]
TRAIN, VAL, INK = "#2775B6", "#CE7643", "#253649"


def load_run(name, run_id, color):
    directory = ROOT / "results/a1/runs" / run_id
    with (directory / "history.csv").open(newline="") as stream:
        history = [{key: float(value) for key, value in row.items()}
                   for row in csv.DictReader(stream)]
    metadata = json.loads((directory / "metrics.json").read_text())
    best_index = max(range(len(history)), key=lambda i: history[i]["val_macro_f1"])
    best = history[best_index]
    minimum = min(history, key=lambda row: row["val_loss"])
    assert [row["epoch"] for row in history] == list(range(len(history)))
    assert metadata["best_epoch"] == best["epoch"]
    assert abs(metadata["best_value"] - best["val_macro_f1"]) < 1e-12
    assert metadata["epochs_run"] == len(history)
    assert len(history) - 1 - best_index == 5
    return dict(name=name, run_id=run_id, color=color, history=history,
                best=best, minimum=minimum, metadata=metadata)


def polish(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#C1CAD2")
    ax.grid(axis="y", color="#DDE4E9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, color="#9CAAB6")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.set_xlabel("Epoch index (starts at 0)")


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, facecolor="white")
    fig.savefig(OUT / f"{name}.svg", facecolor="white")
    plt.close(fig)


def plot_losses(runs):
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 7.5), layout="constrained")
    for letter, ax, run in zip("abcde", axes.flat, runs[:5]):
        rows, best = run["history"], run["best"]
        x = [row["epoch"] for row in rows]
        ax.plot(x, [row["train_loss"] for row in rows], color=TRAIN, linewidth=1.8)
        ax.plot(x, [row["val_loss"] for row in rows], color=VAL, linewidth=1.8)
        ax.axvline(best["epoch"], color=INK, linestyle=":", linewidth=1.1)
        ax.scatter(best["epoch"], best["val_loss"], s=34, color=VAL,
                   edgecolors="white", linewidths=0.8, zorder=4)
        ax.set(title=f"({letter}) {run['name']}", ylabel="Cross-entropy",
               xlim=(-0.4, 27.5), ylim=(0.1, 0.8))
        polish(ax)
    key = axes.flat[-1]
    key.axis("off")
    key.text(0.04, 0.90, "READING THE CURVES", transform=key.transAxes,
             fontsize=12, weight="bold", color=INK)
    key.legend(handles=[
        Line2D([], [], color=TRAIN, linewidth=2, label="Training loss"),
        Line2D([], [], color=VAL, linewidth=2, label="Validation loss"),
        Line2D([], [], color=INK, linestyle=":", label="Selected epoch (best macro-F1)"),
    ], loc="upper left", bbox_to_anchor=(0.00, 0.82), frameon=False,
               handlelength=2.7, labelspacing=0.8, fontsize=11)
    key.text(0.04, 0.29,
             "Shared axes across all five models.\n"
             "Lines end at the recorded stopping epoch.\n"
             "No smoothing or extrapolation. Seed 42.",
             transform=key.transAxes, fontsize=10.5, linespacing=1.4,
             verticalalignment="top", color=INK)
    save(fig, "main_model_losses")


def plot_main_f1(runs):
    fig, ax = plt.subplots(figsize=(10.5, 3.6), layout="constrained")
    for run in runs[:5]:
        rows, best = run["history"], run["best"]
        ax.plot([r["epoch"] for r in rows], [r["val_macro_f1"] for r in rows],
                label=run["name"], color=run["color"], linewidth=2)
        ax.scatter(best["epoch"], best["val_macro_f1"], s=44, color=run["color"],
                   edgecolors="white", linewidths=0.9, zorder=5)
    ax.set(xlim=(-0.4, 27.5), ylim=(0.80, 0.94), ylabel="Validation macro-F1")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    polish(ax)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#DDE4E9",
              ncol=2, fontsize=11)
    save(fig, "main_model_validation_f1")


def plot_lstm(runs):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.8), layout="constrained")
    for run in [runs[3], runs[5], runs[6]]:
        rows, best = run["history"], run["best"]
        x = [r["epoch"] for r in rows]
        label = run["name"].removeprefix("LSTM ").capitalize()
        axes[0].plot(x, [r["val_loss"] for r in rows], color=run["color"],
                     label=label, linewidth=1.9)
        axes[0].plot(x, [r["train_loss"] for r in rows], color=run["color"],
                     linestyle="--", linewidth=1.3, alpha=0.85)
        axes[1].plot(x, [r["val_macro_f1"] for r in rows], color=run["color"],
                     label=label, linewidth=1.9)
        axes[1].scatter(best["epoch"], best["val_macro_f1"], s=38,
                        color=run["color"], edgecolors="white", linewidths=0.8, zorder=5)
    axes[0].set(title="(a) Cross-entropy", ylabel="Loss", ylim=(0.08, 0.83))
    axes[0].legend(loc="upper right", frameon=False, fontsize=11)
    axes[1].set(title="(b) Validation macro-F1", ylabel="Macro-F1", ylim=(0.79, 0.925))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[1].legend(loc="lower right", frameon=False, fontsize=11)
    for ax in axes:
        ax.set_xlim(-0.4, 26.4)
        polish(ax)
    save(fig, "lstm_variant_curves")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 13, "axes.titleweight": "bold",
                         "axes.labelsize": 11, "xtick.labelsize": 11,
                         "ytick.labelsize": 11, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
    runs = [load_run(*spec) for spec in RUNS]
    plot_losses(runs)
    plot_main_f1(runs)
    plot_lstm(runs)
    audit = {"source": "Saved histories and metrics only; no training or inference.",
             "epoch_convention": "zero-based", "runs": []}
    for run in runs:
        rows, best, minimum = run["history"], run["best"], run["minimum"]
        audit["runs"].append(dict(
            model=run["name"], run_id=run["run_id"], seed=run["metadata"]["seed"],
            epochs_run=len(rows), selected_epoch=int(best["epoch"]),
            min_validation_loss_epoch=int(minimum["epoch"]),
            min_validation_loss=minimum["val_loss"],
            best_validation_macro_f1=best["val_macro_f1"],
            selected_train_loss=best["train_loss"], selected_val_loss=best["val_loss"],
            final_epoch=int(rows[-1]["epoch"]), final_train_loss=rows[-1]["train_loss"],
            final_val_loss=rows[-1]["val_loss"],
            final_val_macro_f1=rows[-1]["val_macro_f1"],
            selected_loss_gap=best["val_loss"] - best["train_loss"],
            final_loss_gap=rows[-1]["val_loss"] - rows[-1]["train_loss"],
            non_improving_epochs_after_best=len(rows) - 1 - int(best["epoch"])))
    target = ROOT / "results/a1/tables/a1_training_behaviour.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Wrote three PNG/SVG plot pairs and {target.relative_to(ROOT)}")
    for run in audit["runs"]:
        print(f"{run['model']:14s} epochs={run['epochs_run']:2d} "
              f"best={run['selected_epoch']:2d} min_loss={run['min_validation_loss_epoch']:2d} "
              f"F1={100 * run['best_validation_macro_f1']:.2f}%")


if __name__ == "__main__":
    main()
