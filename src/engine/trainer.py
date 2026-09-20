"""Train/validation loop. Owner: TV 2. Task A1-02.

Handbook 3.2 wants all of this recorded: optimizer, lr, batch size, epochs, scheduler,
regularization, early stopping, checkpoint criterion, seed, hardware, training time.

CONTRACT (agreed 20/09):
    fit(model, loaders, config, run_dir) -> dict with the training history
    - writes history.csv (one row per epoch) into run_dir
    - saves the best checkpoint by config['training']['checkpoint_metric']
"""
from __future__ import annotations

import pathlib

import torch.nn as nn
from torch.utils.data import DataLoader


def fit(model: nn.Module, loaders: tuple[DataLoader, DataLoader, DataLoader],
        config: dict, run_dir: pathlib.Path) -> dict:
    """Train with validation after every epoch.

    TODO (TV 2, A1-02):
      - build optimizer / loss / scheduler from config (CrossEntropyLoss on raw logits)
      - per epoch: train, validate, append to history.csv, keep the best checkpoint
      - early stopping on patience; measure and return total training time
    """
    raise NotImplementedError


def evaluate(model: nn.Module, loader: DataLoader, config: dict) -> dict:
    """Run one pass and return the metric dict from src.engine.metrics. TODO (TV 2, A1-02)."""
    raise NotImplementedError
