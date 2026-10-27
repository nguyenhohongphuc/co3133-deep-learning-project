from __future__ import annotations

import argparse
import time
import json 
import yaml

from src.models import MODELS
from src.data.loaders import build_loaders
from src.engine.trainer import fit
from src.utils.seed import set_seed
from src.utils.run_dir import make_run_dir
from src.utils.env import get_env_info
from src.utils.git_commit import git_commit


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cnn", choices=["linear", "mlp", "cnn", "rnn", "transformer"])
    args = parser.parse_args() 
    return args


def main():
    args = parse_args()
    commit = git_commit()
    with open(f"configs/a1.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    set_seed(config["seed"])
    model_cfg = config["models"][args.model]
    run_name = args.model
    if "sequence" in model_cfg:
        run_name += f"_{model_cfg['sequence']}"
    run_id, run_dir = make_run_dir("a1", run_name, config["seed"])
    with open (run_dir / "config.yaml", "w", encoding = "utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    model = MODELS[args.model](**config["models"][args.model])
    train_loader, val_loader,_ = build_loaders(config)
    loader = {"train": train_loader, "val": val_loader}
    start_time = time.time()
    result = fit(model, loader, config, run_dir)
    train_time = time.time() - start_time
    metrics = {
        "run_id": run_id,
        "model": args.model,
        "seed": config["seed"],
        "commit": commit,
        "device": result["device"],
        "env": get_env_info(result["device"]),
        "param": sum(p.numel() for p in model.parameters()),
        "epochs_run": len(result["history"]),
        "train_time": train_time,
        "best_epoch": result["best_epoch"],
        "best_value": result["best_value"],
        "monitor": config["training"]["monitor"],
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
if __name__ == "__main__":
    main()
