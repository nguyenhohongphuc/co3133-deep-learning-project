# A1 report helpers

This folder contains scripts added for the Assignment 1 report. Run commands from the repository root; outputs remain under `reports/a1/assets/` and `results/a1/tables/`.

| Script | Purpose |
| --- | --- |
| `draw_a1_pipeline.py` | Draw the overall pipeline. |
| `draw_a1_architectures.py` | Draw the five model architectures. |
| `plot_a1_training_behaviour.py` | Plot archived training histories. |
| `summarize_a1_quantitative.py` | Audit saved metrics and plot confusion matrices. |
| `analyze_a1_comparison.py` | Compare saved results, examples, and model costs. |
| `plot_a1_qualitative.py` | Replot verified examples from the saved predictions. |
| `complete_a1_report.py` | Generate final tables, figures, and the input fingerprint. |
| `reproduce_a1.py` | Print a reconstruction plan; training requires explicit `--execute`. |

See `reports/a1/README.md` for dependencies, generation order, and build instructions. For example:

```sh
python scripts/a1_report/plot_a1_training_behaviour.py
python scripts/a1_report/reproduce_a1.py --run-id a1_cnn_s42_1027-0157
```

The existing data, training, evaluation, and A2 utilities stay in their original locations.
