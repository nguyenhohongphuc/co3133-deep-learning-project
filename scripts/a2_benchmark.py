#!/usr/bin/env python3
"""
A2 compute-cost benchmark (proposal item 10, handbook S18 / S20 / S21).

Runs a short real training loop on the actual training folder (real JPEG decoding +
augmentation + forward/backward), then measures inference latency, and extrapolates:
    epoch time, total GPU-hours for the planned experiments, peak GPU memory.

Pretrained weights do not change speed, so weights are NOT downloaded (weights=None).

Usage (Colab: copy the dataset to /content first; reading from Drive is much slower):
    python scripts/a2_benchmark.py --train-dir data/30VNFoods/Train --model resnet50 --amp
    python scripts/a2_benchmark.py --train-dir data/30VNFoods/Train --model resnet50 --amp --freeze
    python scripts/a2_benchmark.py --train-dir data/30VNFoods/Train --model small_cnn --amp

Output: results/a2/compute/benchmark_<model>_<full|frozen>.json (+ printed summary)
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class SmallCNN(nn.Module):
    """From-scratch baseline: 4 conv blocks + global average pooling."""

    def __init__(self, n_cls: int):
        super().__init__()
        layers, c_in = [], 3
        for c_out in (32, 64, 128, 256):
            layers += [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
                       nn.ReLU(inplace=True), nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
                       nn.BatchNorm2d(c_out), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, n_cls)

    def forward(self, x):
        return self.fc(torch.flatten(self.pool(self.features(x)), 1))


def build_model(name: str, n_cls: int, freeze: bool):
    if name == "small_cnn":
        m = SmallCNN(n_cls); head = m.fc
    elif name == "resnet50":
        m = models.resnet50(weights=None); m.fc = nn.Linear(m.fc.in_features, n_cls); head = m.fc
    elif name == "resnet18":
        m = models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, n_cls); head = m.fc
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_cls); head = m.classifier
    elif name == "convnext_tiny":
        m = models.convnext_tiny(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, n_cls); head = m.classifier
    else:
        raise ValueError(name)
    if freeze and name != "small_cnn":
        for p in m.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True
    return m


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-dir", required=True, type=Path)
    ap.add_argument("--model", default="resnet50",
                    choices=["small_cnn", "resnet18", "resnet50", "efficientnet_b0", "convnext_tiny"])
    ap.add_argument("--freeze", action="store_true", help="frozen backbone, train head only")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--amp", action="store_true", help="mixed precision (fp16) on CUDA")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=60, help="timed training iterations")
    ap.add_argument("--eval-images", type=int, default=0,
                    help="images evaluated per epoch (val set size); 0 = 15%% of train")
    ap.add_argument("--epochs", type=int, default=15, help="planned epochs per run")
    ap.add_argument("--runs", type=int, default=3, help="planned runs (e.g. seeds) of this config")
    ap.add_argument("--out", type=Path, default=Path("results/a2/compute"))
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and dev.type == "cuda"
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)

    tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    ds = datasets.ImageFolder(str(args.train_dir), transform=tf)
    n_train, n_cls = len(ds), len(ds.classes)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=dev.type == "cuda", drop_last=True,
                        persistent_workers=args.workers > 0)

    model = build_model(args.model, n_cls, args.freeze).to(dev).to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print(f"Device: {dev} | model {args.model} ({'frozen' if args.freeze else 'full'}) | "
          f"{n_train} train imgs, {n_cls} classes | params {n_params/1e6:.2f}M "
          f"(trainable {n_trainable/1e6:.2f}M)")

    # ---------------- training throughput ----------------
    model.train()
    it = iter(loader)
    data_wait = 0.0
    t0 = None
    for step in range(args.warmup + args.iters):
        if step == args.warmup:
            sync(dev); t0 = time.perf_counter(); data_wait = 0.0
        td = time.perf_counter()
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader); x, y = next(it)
        data_wait += time.perf_counter() - td
        x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(dev, non_blocking=True)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
            loss = crit(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if step % 20 == 0:
            print(f"  step {step:4d} loss {loss.item():.3f}")
    sync(dev)
    elapsed = time.perf_counter() - t0
    train_ips = args.iters * args.batch_size / elapsed
    peak_mem = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else None

    # ---------------- inference latency ----------------
    model.eval()

    def bench(bs, n=30, warm=5):
        x = torch.randn(bs, 3, args.img_size, args.img_size, device=dev).to(memory_format=torch.channels_last)
        times = []
        with torch.inference_mode(), torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
            for i in range(warm + n):
                sync(dev); t = time.perf_counter(); model(x); sync(dev)
                if i >= warm:
                    times.append(time.perf_counter() - t)
        return statistics.median(times)

    lat1 = bench(1)
    latb = bench(args.batch_size, n=15)
    infer_ips = args.batch_size / latb

    # ---------------- extrapolation ----------------
    eval_imgs = args.eval_images or int(0.15 * n_train)
    epoch_train_s = n_train / train_ips
    epoch_eval_s = eval_imgs / infer_ips  # model-only; data loading adds some overhead
    epoch_s = epoch_train_s + epoch_eval_s
    run_h = epoch_s * args.epochs / 3600
    total_h = run_h * args.runs

    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else platform.processor() or "CPU"
    res = {
        "model": args.model, "mode": "frozen" if args.freeze else "full",
        "device": gpu, "amp": use_amp, "img_size": args.img_size, "batch_size": args.batch_size,
        "workers": args.workers, "train_images": n_train, "num_classes": n_cls,
        "params_total": n_params, "params_trainable": n_trainable,
        "train_throughput_img_s": round(train_ips, 1),
        "data_wait_fraction": round(data_wait / elapsed, 3),
        "peak_gpu_mem_GB": round(peak_mem, 2) if peak_mem is not None else None,
        "infer_latency_bs1_ms": round(lat1 * 1000, 2),
        f"infer_latency_bs{args.batch_size}_ms": round(latb * 1000, 2),
        "infer_throughput_img_s": round(infer_ips, 1),
        "est_epoch_train_min": round(epoch_train_s / 60, 2),
        "est_epoch_eval_min": round(epoch_eval_s / 60, 2),
        "planned_epochs": args.epochs, "planned_runs": args.runs,
        "est_hours_per_run": round(run_h, 2), "est_total_gpu_hours": round(total_h, 2),
        "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                     "python": platform.python_version(),
                     "cuda": torch.version.cuda if dev.type == "cuda" else None},
        "timed_iters": args.iters, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    f = args.out / f"benchmark_{args.model}_{res['mode']}.json"
    f.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print("\n=== Compute estimate ===")
    print(f"Device {gpu}, AMP={use_amp}, batch {args.batch_size}, {args.img_size}px")
    print(f"Train throughput: {train_ips:.1f} img/s (data wait {res['data_wait_fraction']*100:.0f}% of time)")
    if peak_mem is not None:
        print(f"Peak GPU memory: {peak_mem:.2f} GB")
    print(f"Inference: {lat1*1000:.1f} ms/img (bs=1), {infer_ips:.0f} img/s (bs={args.batch_size})")
    print(f"Epoch ~ {epoch_s/60:.1f} min (train {epoch_train_s/60:.1f} + eval {epoch_eval_s/60:.1f})")
    print(f"{args.epochs} epochs x {args.runs} runs ~ {total_h:.2f} GPU-hours")
    if res["data_wait_fraction"] > 0.3:
        print("NOTE: >30% time waiting for data -> CPU/disk bound. Try more --workers, "
              "or pre-resize images (e.g. shorter side 256) to speed up training.")
    print(f"Saved -> {f}")


if __name__ == "__main__":
    main()
