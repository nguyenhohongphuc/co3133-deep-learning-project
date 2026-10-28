# A1 report

The editable project is `main.tex` in this directory. The repository-root `main.tex` wraps the same document. `main.pdf` is the editor preview output; `NNs_A1_Report.pdf` is an identical submission export named according to the revised handbook.

## Build

Run from `reports/a1/` with Tectonic available:

```sh
tectonic --synctex --keep-logs main.tex
cp main.pdf NNs_A1_Report.pdf
```

An offline build may add `--only-cached` after the required fonts/packages have been cached. The built-in editor compiler checks the same project but may not write the on-disk preview, so the final command updates that file explicitly.

## Regenerate evidence

From the repository root, use an environment with NumPy, pandas, matplotlib, Pillow, PyYAML, and the project's standard dependencies. `analyze_a1_comparison.py` also uses local Tesseract to read existing plot text. Raw Fashion-MNIST IDX files and the archived run results must be present.

```sh
python scripts/a1_report/draw_a1_pipeline.py
python scripts/a1_report/draw_a1_architectures.py
python scripts/a1_report/plot_a1_training_behaviour.py
python scripts/a1_report/summarize_a1_quantitative.py
python scripts/a1_report/analyze_a1_comparison.py
python scripts/a1_report/plot_a1_qualitative.py
python scripts/a1_report/complete_a1_report.py
```

The comparison audit precedes the qualitative plot so all transcribed CNN annotations are cross-checked. These commands do not train a model or run checkpoint inference. On machines with slow system-font discovery, use matplotlib's bundled fonts with `MPL_IGNORE_SYSTEM_FONTS=1`, and set `MPLCONFIGDIR` to a writable cache directory.

The completion script writes `results/a1/tables/a1_reproducibility_manifest.json`, including full input hashes and weight availability. Run it after changing report sources to refresh the fingerprint.

## Reconstruct a model

All seven saved checkpoints are tracked at `results/a1/runs/<run_id>/best.pt`. The trainer saves model `state_dict` values (parameters and buffers), not optimizer or RNG states. File presence, archive integrity, and SHA-256 hashes were checked. Loading into models and test-metric reproduction were not verified in this update.

To re-evaluate an existing checkpoint from the repository root:

```sh
python -m src.evaluate --run_dir results/a1/runs/a1_cnn_s42_1027-0157
```

Select another archived run ID as needed. Use a disposable copy: the evaluator overwrites `test_metrics.json` and test figures. It does not retrain the model.

`scripts/a1_report/reproduce_a1.py --run-id <archived-run-id>` prints a plan without changes. Adding `--execute` explicitly trains and evaluates a new run using the archived snapshot, restoring the central configuration afterward. Use a disposable copy and a prepared training environment. Original artifacts are not overwritten. The execution path was not run during report preparation.

## Points for the requested review

- Seven saved runs use seed 42; no multi-seed statistics are claimed.
- Seven saved checkpoints are available. Two LSTM runs have dirty training states, optimizer/RNG states are not saved, and the exported dependency snapshot needs reconciliation. These limits and evaluation/retraining instructions are documented.
- Test #32 is **Dress predicted as Trouser**. An earlier transcription error was corrected against the saved grid and annotation audit.
- Member roles are supported by repository history and the AI log. The group should verify that summary and disclosure before submission.

The draft includes report sources, helpers, analysis artifacts, and the named PDF export for group review. Pushing this draft does not submit the report to the LMS.
