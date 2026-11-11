"""Evaluate all available A2 checkpoints, with a fixed validation-based selection rule.

    python scripts/a2_evaluate_all.py --split test --device auto --batch-size 32

Existing complete exports are reused only when checkpoint/config/split hashes and
evaluation settings match. --rerun explicitly replaces matching existing exports.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml
from src.a2.evaluate import evaluate_run, parse_args as evaluation_args


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', required=True, choices=['val', 'test'])
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--threads', default=5, type=int)
    parser.add_argument('--rerun', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print('DEVICE', device, 'SPLIT', args.split, flush=True)
    runs = sorted(p.parent for p in (ROOT / 'results/a2/runs').glob('*/best.pt'))
    if not runs:
        raise SystemExit('No A2 best.pt checkpoints found')
    tables = ROOT / 'results/a2/tables'
    tables.mkdir(parents=True, exist_ok=True)

    # Choose illustrative runs before reading any test metric; never pick a seed by test score.
    representatives = {}
    for run in runs:
        validation = run / 'evaluation/val/metrics.json'
        if not validation.is_file():
            if args.split == 'test':
                raise SystemExit(f'Validation result required before test selection: {run.name}')
            continue
        val = read_json(validation)
        previous = representatives.get(val['model'])
        if previous is None or val['macro_f1'] > previous['validation_macro_f1']:
            representatives[val['model']] = dict(run_id=run.name, validation_macro_f1=val['macro_f1'])
    protocol = dict(split=args.split, run_ids=[p.name for p in runs], device=device,
                    batch_size=args.batch_size, cpu_threads=args.threads, num_workers=0,
                    representatives=representatives,
                    selection_rule='Highest existing validation macro-F1 within each model family; fixed before test',
                    checkpoint_sha256={run.name: digest(run / 'best.pt') for run in runs})
    protocol_path = tables / f'a2_{args.split}_protocol.json'
    if protocol_path.exists() and read_json(protocol_path)['representatives'] != representatives:
        raise SystemExit('Representative selection differs from the existing protocol; review it explicitly')
    protocol_path.write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')

    current = {}
    stop = threading.Event()
    def heartbeat():
        while not stop.wait(20):
            run = current.get('run')
            if run is None:
                continue
            paths = list((run / 'evaluation' / args.split).glob('.a2-eval-*/predictions.csv'))
            count = 0
            try:
                if paths:
                    with paths[0].open(encoding='utf-8') as stream:
                        count = max(0, sum(1 for _ in stream) - 1)
            except FileNotFoundError:
                continue
            print('PROGRESS', run.name, count, 'rows', round(time.monotonic()-current['start']), 'seconds', flush=True)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    rows = []
    try:
        for index, run in enumerate(runs, 1):
            current.update(run=run, start=time.monotonic())
            print('START', index, '/', len(runs), run.name, flush=True)
            out = run / 'evaluation' / args.split
            files = ['metrics.json', 'predictions.csv', 'per_class_metrics.csv', 'confusion_matrix.csv']
            reuse = False
            if not args.rerun and all((out / name).is_file() for name in files):
                m = read_json(out / 'metrics.json')
                cfg = yaml.safe_load((run / 'config.yaml').read_text())
                split_path = Path(cfg['split_file'])
                if not split_path.is_absolute():
                    split_path = ROOT / split_path
                reuse = (m['checkpoint_sha256'] == protocol['checkpoint_sha256'][run.name]
                         and m['config_sha256'] == digest(run/'config.yaml')
                         and m['split_sha256'] == digest(split_path)
                         and m['device'] == device and m['batch_size'] == args.batch_size
                         and m['num_workers'] == 0)
                if reuse:
                    with (out / 'predictions.csv').open(encoding='utf-8') as stream:
                        reuse = sum(1 for _ in csv.DictReader(stream)) == m['num_samples']
                if not reuse:
                    raise SystemExit(f'Existing export/settings mismatch for {run.name}; inspect before using --rerun')
            if not reuse:
                m, out = evaluate_run(evaluation_args([
                    '--run-dir', str(run), '--split', args.split, '--device', device,
                    '--batch-size', str(args.batch_size), '--num-workers', '0']))
            row = {key: m[key] for key in ['run_id', 'model', 'seed', 'split', 'num_samples',
                                          'accuracy', 'macro_f1', 'weighted_f1', 'top3_accuracy', 'loss',
                                          'device', 'batch_size', 'inference_ms_per_image',
                                          'inference_images_per_second', 'evaluation_seconds', 'checkpoint_sha256']}
            row['predictions_file'] = (out/'predictions.csv').relative_to(ROOT).as_posix()
            row['representative'] = representatives.get(m['model'], {}).get('run_id') == run.name
            rows.append(row)
            with (tables/f'a2_{args.split}_evaluation.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
            print('DONE', json.dumps(row), flush=True)
            current.clear()
            gc.collect()
            if device == 'mps':
                torch.mps.empty_cache()
        summary = {}
        for model in sorted({row['model'] for row in rows}):
            group = [row for row in rows if row['model'] == model]
            summary[model] = dict(runs=len(group), seeds=[row['seed'] for row in group])
            for metric in ['accuracy', 'macro_f1', 'top3_accuracy', 'inference_ms_per_image']:
                values = [row[metric] for row in group]
                summary[model][metric+'_mean'] = statistics.mean(values)
                summary[model][metric+'_sample_std'] = statistics.stdev(values) if len(values)>1 else None
        (tables/f'a2_{args.split}_summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
        print('COMPLETE', json.dumps(summary), flush=True)
    finally:
        stop.set()
        thread.join(timeout=1)


if __name__ == '__main__':
    main()
