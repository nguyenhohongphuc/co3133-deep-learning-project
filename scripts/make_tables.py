from __future__ import annotations

import argparse
import json
import pathlib

import pandas as pd
import yaml

MODEL_ORDER = ["linear", "mlp", "cnn", "rnn", "transformer"]


def load_runs(runs_dir: pathlib.Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(runs_dir.iterdir()):
        metrics_path = run_dir / "metrics.json"
        test_path = run_dir / "test_metrics.json"
        if not metrics_path.exists() or not test_path.exists():
            print(f"skip {run_dir.name}: not trained or not evaluated yet")
            continue
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        with open(test_path, encoding="utf-8") as f:
            test = json.load(f)
        with open(run_dir / "config.yaml", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        rows.append({
            "run_id": metrics["run_id"],
            "model": metrics["model"],
            # rows / cols / patches for rnn and transformer (A1-15 ablation), "-" otherwise
            "sequence": config["models"][metrics["model"]].get("sequence", "-"),
            "seed": metrics["seed"],
            "test_accuracy": test["accuracy"],
            "test_macro_f1": test["macro_f1"],
            "params": metrics["param"],
            "train_time_s": metrics["train_time"],
            "inference_time_s": test["inference_time"],
            "epochs_run": metrics["epochs_run"],
            "best_epoch": metrics["best_epoch"],
            "device": metrics["device"],
            "commit": metrics["commit"],
        })
    return pd.DataFrame(rows)


def check_fairness(df: pd.DataFrame) -> None:
    """Handbook 12.2: every model on the same device, every run traceable to a commit."""
    if df["device"].nunique() > 1:
        print(f"WARNING: runs use different devices {sorted(df['device'].unique())}; times are not comparable")
    dirty = df[df["commit"].str.endswith("-dirty")]
    for run_id in dirty["run_id"]:
        print(f"WARNING: {run_id} was trained with uncommitted changes")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """mean +/- std over seeds; std is NaN when a model has only one seed."""
    cols = ["test_accuracy", "test_macro_f1", "train_time_s", "inference_time_s"]
    groups = df.groupby(["model", "sequence"])
    summary = groups[cols].agg(["mean", "std"])
    summary.columns = [f"{c}_{stat}" for c, stat in summary.columns]
    summary["params"] = groups["params"].first()
    summary["n_seeds"] = groups["seed"].nunique()
    order = sorted(summary.index, key=lambda k: (MODEL_ORDER.index(k[0]), k[1]))
    return summary.loc[order]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--assignment", default="a1", choices=["a1", "a2"])
    args = p.parse_args()

    root = pathlib.Path("results") / args.assignment
    df = load_runs(root / "runs")
    if df.empty:
        raise SystemExit("no evaluated runs found")

    # If a model+sequence+seed was run more than once, keep the latest run (run_id ends with MMDD-HHMM).
    df = df.sort_values("run_id").drop_duplicates(["model", "sequence", "seed"], keep="last")
    check_fairness(df)

    out_dir = root / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "comparison.csv", index=False)
    summary = summarize(df)
    summary.to_csv(out_dir / "comparison_summary.csv")

    print(summary.round(4).to_string())
    print(f"\nWrote {out_dir / 'comparison.csv'} and {out_dir / 'comparison_summary.csv'}")


if __name__ == "__main__":
    main()