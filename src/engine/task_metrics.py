"""A2 task metrics. Owner: TV 2. Task A2-07 / A2-10.

Handbook 20.1: classification = accuracy + macro-F1; detection = mAP (+ AP by class/size);
segmentation = mIoU + Dice (+ mask AP); Re-ID = Rank-1, Rank-5, mAP;
depth = Abs Rel, RMSE, delta accuracy; text tasks = justify the metric explicitly.
"""
from __future__ import annotations


def build_task_metrics(task: str):
    """Return a callable(y_true, y_pred) -> dict for the approved track. TODO (TV 2)."""
    raise NotImplementedError
