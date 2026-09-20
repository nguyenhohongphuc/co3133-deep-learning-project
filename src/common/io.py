"""Run bookkeeping: run ids, config copies, git commit hash.

Owner: TV 2. Every reported number must be traceable to config + split + checkpoint
+ run id + commit (handbook 4.2), and that is exactly what a run folder stores.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
from datetime import datetime
from typing import Any

RESULTS = pathlib.Path("results")


def git_commit() -> str:
    """Short commit hash, or 'nogit' when the command fails."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "nogit"


def make_run_id(run_name: str, seed: int) -> str:
    """a1_cnn + seed 42 -> 'a1_cnn_s42_0312-1745'."""
    return f"{run_name}_s{seed}_{datetime.now():%m%d-%H%M}"


def run_dir(assignment: str, run_id: str, create: bool = True) -> pathlib.Path:
    """results/<assignment>/runs/<run_id>/ - holds config.yaml, history.csv, metrics.json."""
    path = RESULTS / assignment / "runs" / run_id
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
