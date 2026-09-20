"""Build the comparison table from the run folders. Owner: TV 2. Task A1-16.

    python scripts/make_tables.py --assignment a1

Reads results/a1/runs/*/metrics.json and writes results/a1/tables/comparison.csv with
accuracy, macro-F1, parameters, training time and inference time for all five models
(handbook 12.1), plus mean +/- std when several seeds exist (3.3).
"""
from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--assignment", default="a1", choices=["a1", "a2"])
    args = p.parse_args()
    raise NotImplementedError  # TODO (TV 2, A1-16)


if __name__ == "__main__":
    main()
