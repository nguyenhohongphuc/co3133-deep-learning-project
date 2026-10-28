"""Plan or rerun one archived A1 configuration using the current entry points.

Default: read-only plan. --execute explicitly trains a NEW run and evaluates it;
the original archived run is never overwritten. Use a disposable workspace copy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Archived run folder name")
    parser.add_argument("--execute", action="store_true", help="Actually train and evaluate a new run")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        parser.error("--run-id must be a folder name, not a path")
    archive = ROOT / "results/a1/runs" / args.run_id
    metadata = json.loads((archive / "metrics.json").read_text())
    snapshot = (archive / "config.yaml").read_bytes()
    import yaml
    config = yaml.safe_load(snapshot)
    model = metadata["model"]
    if model not in {"linear", "mlp", "cnn", "rnn", "transformer"}:
        parser.error("Unsupported archived model")
    assert config["seed"] == metadata["seed"]
    assert (ROOT / config["split_file"]).is_file()
    command = [sys.executable, "-m", "src.train", "--model", model]
    print("Source snapshot:", args.run_id)
    print("Model:", model, "seed:", config["seed"], "sequence:", config["models"][model].get("sequence", "-"))
    print("Recorded training revision:", metadata["commit"])
    print("Current revision:", subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip())
    print("Training command:", " ".join(command))
    if not args.execute:
        print("Plan only: no config writes, training, evaluation, or new run directory.")
        return
    central = ROOT / "configs/a1.yaml"
    original = central.read_bytes()
    before = {p.name for p in (ROOT / "results/a1/runs").iterdir() if p.is_dir()}
    try:
        # This may legitimately mark configs/ dirty; new metadata must retain it.
        if yaml.safe_load(original) != config:
            central.write_bytes(snapshot)
        subprocess.run(command, cwd=ROOT, check=True)
        candidates = []
        for directory in (ROOT / "results/a1/runs").iterdir():
            if directory.is_dir() and directory.name not in before and (directory / "metrics.json").is_file():
                fresh = json.loads((directory / "metrics.json").read_text())
                if fresh["model"] == model:
                    candidates.append(directory)
        if len(candidates) != 1:
            raise RuntimeError("Cannot identify exactly one new successful run; evaluate it explicitly")
        new_run = candidates[0]
        subprocess.run([sys.executable, "-m", "src.evaluate", "--run_dir",
                        str(new_run.relative_to(ROOT))], cwd=ROOT, check=True)
        print("New results:", new_run)
        print("Archived results remain at:", archive)
    finally:
        if central.read_bytes() != original:
            central.write_bytes(original)


if __name__ == "__main__":
    main()
