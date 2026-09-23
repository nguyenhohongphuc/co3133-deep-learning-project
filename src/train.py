"""Train one model from one config file. Owner: TV 2.

    python -m src.train --config configs/a1/cnn.yaml
    python -m src.train --config configs/a1/cnn.yaml --seed 1

Writes results/<assignment>/runs/<run_id>/{config.yaml, history.csv, metrics.json}
so every number stays traceable (handbook 4.2).

metrics.json holds the TRAINING half of the record (params, epochs, training time,
losses). Accuracy, macro-F1 and inference time are filled in by src/evaluate.py
(A1-11), which reads last.pt from the same run folder.
"""
from __future__ import annotations

import argparse
import sys
import time

import yaml

import src.models  # noqa: F401  - importing the package registers every model
from src.common.io import git_commit, make_run_id, run_dir, save_json
from src.common.registry import build_model
from src.common.seed import set_seed
from src.data.loaders import build_loaders
from src.engine.trainer import fit

def parse_args():
    p = argparse.ArgumentParser(description="Train one model from a config file")
    p.add_argument("--config", required=True, help="e.g. configs/a1/cnn.yaml")
    p.add_argument("--seed", type=int, default=None, help="override config seed")
    p.add_argument("--epochs", type=int, default=None, help="override for a quick smoke test")
    return p.parse_args()
def load_config(path):
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    parent = config.pop("defaults", None)
    if parent is None:
        return config
    base = load_config(parent)
    for key, value in config.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key].update(value)     # merge inside the block, don't replace it
        else:
            base[key] = value
    return base

def main() -> None:
    # The Windows console is cp1252 by default and cannot print Vietnamese.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()

    # 1. Đọc config đúng cách (kế thừa được từ file gốc)
    config = load_config(args.config)

    # 2. Ghi đè seed và epochs nếu người dùng nhập tham số từ terminal
    if args.seed is not None:
        config["seed"] = args.seed
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    # 3. Cố định seed ngay trước khi tạo model/data
    set_seed(config["seed"])

    # 4. Tạo đường dẫn thư mục lưu kết quả (ví dụ: results/a1/runs/cnn_seed1)
    run_id = make_run_id(config["run_name"], config["seed"])
    rd = run_dir("a1", run_id)

    # 5. Chuẩn bị DataLoader & Model
    train_loader, val_loader, test_loader = build_loaders(config)
    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    model = build_model(config)

    # 6. Lưu lại bản sao file config vào thư mục run để tiện kiểm tra sau này
    (rd / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    # 7. Bắt đầu train & đo thời gian
    start_time = time.perf_counter()
    result = fit(model, loaders, config, run_dir=rd)
    train_time = time.perf_counter() - start_time

    # 8. Lưu các thông số chính ra file metrics.json
    save_json(
        rd / "metrics.json",
        {
            "run_id": run_id,
            "model": config["model"]["name"],
            "seed": config["seed"],
            "commit": git_commit(),
            "device": str(result["device"]),
            "params": sum(p.numel() for p in model.parameters()),
            "epochs_run": len(result["history"]),
            "train_time_sec": round(train_time, 2),
            "best_epoch": result["best_epoch"],
            "best_val_macro_f1": result["best_value"],
        },
    )

    print(f"\n---> Huấn luyện hoàn tất! Tất cả file kết quả đã lưu tại: {rd}")
if __name__ == "__main__":
    main()
