"""Figures for the report and the web page. Owner: TV 2. Task A1-11.

Handbook 12.1: training and validation curves, confusion matrix, and examples of
correct and incorrect predictions. Save everything under results/<a1|a2>/figures/.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_curves(history_csv, out_png):
    df = pd.read_csv(history_csv)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(df["epoch"], df["train_loss"], label="train", color="#4269D0")
    ax[0].plot(df["epoch"], df["val_loss"], label="validation", color="#EFB118")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
    ax[1].plot(df["epoch"], df["val_accuracy"], label="accuracy", color="#4269D0")
    ax[1].plot(df["epoch"], df["val_macro_f1"], label="macro-F1", color="#EFB118")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("score"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def plot_confusion(matrix, class_names, out_png):
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues")          # một tông màu, từ sáng -> tối, tuyệt đối không dùng dải cầu vồng
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    threshold = matrix.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, matrix[i, j], ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black", fontsize=8)
    fig.colorbar(im); fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)

def plot_predictions(images, y_true, y_pred, class_names, out_png, wrong_only=False):
    idx = np.where(y_true != y_pred)[0] if wrong_only else np.arange(len(y_true))
    idx = idx[:16]
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(images[i].squeeze(), cmap="gray")
        ax.set_title(f"true {class_names[y_true[i]]}\npred {class_names[y_pred[i]]}", fontsize=8)
        ax.axis("off")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
