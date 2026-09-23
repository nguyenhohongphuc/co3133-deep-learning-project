"""Parameter count and timing. Owner: TV 2. Task A1-11.

Handbook 12.1 requires number of parameters, training time and inference time for
every model - measured on the same device, batch size and warm-up so the comparison
is fair (12.2).
"""
from __future__ import annotations

import time

import torch

def count_parameters(model, trainable_only=True):
    params = [p for p in model.parameters() if p.requires_grad or not trainable_only]
    return sum(p.numel() for p in params)


def inference_time(model, loader, device, warmup_batches=5):
    model = model.to(device).eval()
    times, batch_size = [], loader.batch_size
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            images = images.to(device)
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(images)
            if device == "cuda":
                torch.cuda.synchronize()
            if i >= warmup_batches:             # bỏ qua các batch khởi động
                times.append(time.perf_counter() - start)
    if not times:
        raise ValueError(f"loader has {warmup_batches + 1} batches or fewer; nothing left to time")
    ms = 1000 * sum(times) / len(times)
    return {"ms_per_batch": ms, "ms_per_image": ms / batch_size,
            "batch_size": batch_size, "device": device}