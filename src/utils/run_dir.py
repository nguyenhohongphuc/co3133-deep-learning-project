from datetime import datetime
from pathlib import Path
import yaml


def make_run_dir(assignment, model_name, seed, root="results"):
    stamp = datetime.now().strftime("%m%d-%H%M")
    run_id = f"{assignment}_{model_name}_s{seed}_{stamp}"
    run_dir = Path(root) / assignment / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir