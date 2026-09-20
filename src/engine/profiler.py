"""Parameter count and timing. Owner: TV 2. Task A1-11.

Handbook 12.1 requires number of parameters, training time and inference time for
every model - measured on the same device, batch size and warm-up so the comparison
is fair (12.2).
"""
from __future__ import annotations

import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """TODO (TV 2, A1-11)."""
    raise NotImplementedError


def inference_time(model: nn.Module, loader, device: str, warmup_batches: int = 5) -> dict:
    """{'ms_per_batch': ..., 'ms_per_image': ..., 'batch_size': ..., 'device': ...}.

    TODO (TV 2, A1-11): warm up first, then time with torch.cuda.synchronize() on GPU.
    """
    raise NotImplementedError
