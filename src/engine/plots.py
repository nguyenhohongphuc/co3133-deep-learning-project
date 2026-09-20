"""Figures for the report and the web page. Owner: TV 2. Task A1-11.

Handbook 12.1: training and validation curves, confusion matrix, and examples of
correct and incorrect predictions. Save everything under results/<a1|a2>/figures/.
"""
from __future__ import annotations

import pathlib


def plot_curves(history_csv: pathlib.Path, out_png: pathlib.Path) -> None:
    """Train/val loss and metric per epoch. TODO (TV 2, A1-11)."""
    raise NotImplementedError


def plot_confusion(matrix, class_names: list[str], out_png: pathlib.Path) -> None:
    """Annotated confusion matrix. TODO (TV 2, A1-11)."""
    raise NotImplementedError


def plot_predictions(images, y_true, y_pred, class_names: list[str],
                     out_png: pathlib.Path, wrong_only: bool = False) -> None:
    """Grid of predictions; wrong_only=True gives the failure cases. TODO (TV 2, A1-11)."""
    raise NotImplementedError
