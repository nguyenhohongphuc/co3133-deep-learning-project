"""Metrics. Owner: TV 2. Task A1-05.

Handbook 12.1 requires accuracy, macro-F1 and a confusion matrix for every A1 model.

Pure numpy in, plain numbers out: no torch, no matplotlib, no file writing. trainer.py
calls macro_f1 once per epoch for the checkpoint rule, so this must stay import-light.
The figures live in src/engine/plots.py (A1-11).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score  # cite in the report (handbook 4.2)


def _check(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten to 1-D int arrays and refuse the two mistakes that fail silently."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true {y_true.shape} and y_pred {y_pred.shape} differ in length")
    if y_true.size == 0:
        raise ValueError("no samples to score")
    return y_true.astype(int), y_pred.astype(int)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correct predictions, in [0, 1]."""
    y_true, y_pred = _check(y_true, y_pred)
    return float((y_true == y_pred).mean())


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Unweighted mean of the per-class F1, so every class counts the same."""
    y_true, y_pred = _check(y_true, y_pred)
    # zero_division=0: a class the model never predicts scores 0, it is not an error.
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def confusion(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """[num_classes, num_classes] counts, rows = truth, columns = prediction."""
    y_true, y_pred = _check(y_true, y_pred)
    if y_true.max(initial=0) >= num_classes or y_pred.max(initial=0) >= num_classes:
        raise ValueError(f"labels outside [0, {num_classes - 1}]")
    flat = np.bincount(y_true * num_classes + y_pred, minlength=num_classes ** 2)
    return flat.reshape(num_classes, num_classes)


def summarize(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    """{'accuracy': ..., 'macro_f1': ..., 'confusion': [[...]]} -> metrics.json."""
    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "confusion": confusion(y_true, y_pred, num_classes).tolist(),
    }
