# Assignment 2 report

The [final report PDF](NNs_A2_Report.pdf) is built from [the LaTeX source](main.tex). The sources below document the data, methodology, and results.

| Report part | Supporting material |
| --- | --- |
| Problem and Data Description | [Proposal](proposal.md), [approval record](approval.md), [EDA notebook](../../notebooks/a2_eda.ipynb), [cleaned split](../../results/a2/splits/split_v2.json) |
| Methodology | [Base config](../../configs/a2.yaml), [model and training code](../../src/a2/), [saved run configurations](../../results/a2/runs/) |
| Implementation Results | [Test analysis](test_evaluation_analysis.md), [result tables](../../results/a2/tables/), [test figures](../../results/a2/figures/test/), [qualitative review](../../results/a2/error_analysis/qualitative_review.csv) |

## Handbook coverage

| [Handbook](../../handbook-ene-v2.pdf) | Where to cover it in the report | Evidence status |
| --- | --- | --- |
| §3.1, §18, §19.1, §22 — problem, proposal, dataset size, EDA, split, preprocessing | Sections 1–3 and Appendix A | Proposal, EDA, split and example batch are available. The approval record needs the group's confirmed decision. |
| §3.2, §20–21 — pipeline, baseline, pretrained method and fine-tuning protocol | Sections 4–7 | Code, resolved configs and run logs are available. Use the saved run configs for reported settings. |
| §3.3, §20–21 — training behavior, accuracy, macro-F1, compute and controlled experiment | Sections 8–9, 11 and 13 | Seven checkpoints were evaluated. Frozen vs. full fine-tuning is the controlled experiment, with three paired seeds. The scratch CNN is a single-run baseline. |
| §3.3, §20 — qualitative examples, error analysis and conclusions | Sections 10, 12 and 14–15 | Includes test examples, Grad-CAM, error frequencies, and analysis of remaining confusions. |
| §4–5 — reproducibility, proposal approval and disclosure | Section 16, Appendix A and AI Usage Disclosure | Includes the run and checkpoint records and the [AI usage log](../../AI_USAGE.md). The instructor's proposal decision remains unverified. |

## Build the report

From `reports/a2/`, build with Tectonic.

```bash
tectonic main.tex
mv main.pdf NNs_A2_Report.pdf
```
