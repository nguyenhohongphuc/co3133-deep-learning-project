"""Train one model from one config file. Owner: TV 2.

    python -m src.train --config configs/a1/cnn.yaml
    python -m src.train --config configs/a1/cnn.yaml --seed 1

Writes results/<assignment>/runs/<run_id>/{config.yaml, history.csv, metrics.json}
so every number stays traceable (handbook 4.2).
"""
from __future__ import annotations

import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Train one model from a config file")
    p.add_argument("--config", required=True, help="e.g. configs/a1/cnn.yaml")
    p.add_argument("--seed", type=int, default=None, help="override config seed")
    p.add_argument("--epochs", type=int, default=None, help="override for a quick smoke test")
    return p.parse_args()


def main() -> None:
    """TODO (TV 2, A1-02):
      load_config -> set_seed -> build_loaders -> build_model -> fit -> save metrics.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
