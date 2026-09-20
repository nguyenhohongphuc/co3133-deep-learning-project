"""Metrics. Owner: TV 2. Task A1-05.

Handbook 12.1 requires accuracy, macro-F1 and a confusion matrix for every A1 model.
"""
from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """TODO (TV 2, A1-05)."""
    raise NotImplementedError


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """TODO (TV 2, A1-05). sklearn.metrics.f1_score(average='macro') is fine - cite it in the report."""
    raise NotImplementedError


def confusion(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """TODO (TV 2, A1-05): [num_classes, num_classes] counts, rows = truth."""
    raise NotImplementedError


def summarize(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    """{'accuracy': ..., 'macro_f1': ..., 'confusion': [[...]]} -> metrics.json."""
    raise NotImplementedError
