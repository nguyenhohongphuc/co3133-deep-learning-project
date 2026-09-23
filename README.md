# CO3133 — Deep Learning and Its Applications — Course Project (Semester 261)

Group **NNs** · Faculty of Computer Science and Engineering · Ho Chi Minh City University of Technology (HCMUT), VNU-HCM
Instructor: Lê Thành Sách · LMS: <https://lms.hcmut.edu.vn/course/view.php?id=142848>

Course website (GitHub Pages): **<https://nguyenhohongphuc.github.io/co3133-deep-learning-project/>**
Repository: **<https://github.com/nguyenhohongphuc/co3133-deep-learning-project>**

| Assignment | Topic | Weight | Page |
|---|---|---|---|
| 1 | Foundations of Deep Learning Pipelines and Architectures | 40% | `docs/assignments/assignment-1.html` |
| 2 | Deep Learning on Large-Scale Data and Specialized Tasks | 30% | `docs/assignments/assignment-2.html` |
| 3 | Multimodal Deep Learning | 30% | `docs/assignments/assignment-3.html` |

## Group members

| Full name | Student ID | Main role | GitHub |
|---|---|---|---|
| Nguyễn Nhật Nam | 2352778 | [role] | <https://github.com/nnam128> |
| Hồ Hồng Phúc Nguyên | 2352824 | [role] | <https://github.com/nguyenhohongphuc> |
| Phan Quốc Đại Sơn | 2353053 | [role] | <https://github.com/sonphan-02> |

> Do not invent GitHub profile links. Leave the cell blank if a member has no profile.

## Repository layout

```
configs/a1|a2/            One YAML per reported run (base.yaml holds the shared settings)
src/
  common/                 seed, run ids + git hash, model registry          (TV 2)
  data/                   split, transforms, Dataset, DataLoader            (TV 1)
  engine/                 trainer, metrics, plots, profiler                 (TV 2)
  models/                 linear, mlp (TV 1) · cnn (TV 2) · rnn, transformer (TV 3)
  train.py evaluate.py    Entry points: python -m src.train --config ...
scripts/                  prepare_data.py, run_a1_all.ps1, make_tables.py
notebooks/                EDA and error analysis only (never the pipeline)
results/a1|a2/            COMMITTED: splits/, runs/<run_id>/, figures/, tables/
reports/a1|a2/            Report sources, slides, A2 proposal + approval status
docs/                     GitHub Pages site (published from /docs on main)
  index.html              Shared landing page
  assignments/            One page per assignment
  assets/css/style.css    Site stylesheet
planning/                 Team task sheet + task guidelines (not a deliverable)
data/ checkpoints/ runs/  Ignored by git; see "Dataset preparation" below
AI_USAGE.md               Detailed AI usage log (mandatory)
```

### Who owns what

| Area | Owner | Files |
|---|---|---|
| Data pipeline, Linear + MLP | TV 1 | `src/data/**`, `src/models/linear.py`, `src/models/mlp.py` |
| Training engine, metrics, plots, CNN | TV 2 | `src/engine/**`, `src/common/**`, `src/train.py`, `src/evaluate.py`, `src/models/cnn.py` |
| LSTM/GRU, Transformer | TV 3 | `src/models/rnn.py`, `src/models/transformer.py` |

Two interfaces are fixed so the three areas stay independent:
`build_loaders(config) -> (train, val, test)` yielding `(images [B,1,28,28], labels [B])`,
and every model is an `nn.Module` with `forward(x) -> logits [B, num_classes]`.
Each model registers itself with `@register("name")` from `src/common/registry.py`.

## Publishing the site (one-time setup)

1. Push this repository to GitHub (public).
2. **Settings → Pages → Build and deployment**: source *Deploy from a branch*, branch `main`, folder `/docs`.
3. Wait for the first deployment, open the published URL, and paste it into the table above and into the LMS submission.

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Pinned dependency versions live in `requirements.txt` (TODO: create it once the environment is fixed).

## Dataset preparation

```bash
python scripts/prepare_data.py --dataset fashion-mnist --out data/   # primary dataset
python scripts/prepare_data.py --dataset mnist --out data/           # debugging only
```

The fixed split is built once and then reused by every model:

```bash
python -m src.data.split --config configs/a1/base.yaml   # writes results/a1/splits/split_seed42.json
```

## Train

```bash
python -m src.train --config configs/a1/linear.yaml
python -m src.train --config configs/a1/cnn.yaml --seed 1

./scripts/run_a1_all.ps1      # all five models, one split, one seed
./scripts/run_a1_seeds.ps1    # the same comparison over 3 seeds (mean ± std)
```

## Evaluate

```bash
python -m src.evaluate --config configs/a1/cnn.yaml \
    --checkpoint results/a1/runs/<run_id>/best.pt --split test
python scripts/make_tables.py --assignment a1   # results/a1/tables/comparison.csv
```

Each run writes `results/a1/runs/<run_id>/` containing `config.yaml` (the *resolved* config,
with any `--seed` / `--epochs` override applied), `history.csv`, `metrics.json` and `best.pt`,
where `run_id = <run_name>_s<seed>_<MMDD-HHMM>`.

`src/train.py` never touches the test set: it trains on the train split and selects the
checkpoint on validation macro-F1. `src/evaluate.py --split test` is the only code that reads
the test set, and it is meant to be run once per model, at the end. It writes
`metrics_test.json`, folds accuracy, macro-F1 and inference time back into `metrics.json`, and
saves the handbook §12.1 figures to `results/a1/figures/`.

## Reproducibility

Every reported number must be traceable to its configuration, dataset split, checkpoint, log / experiment ID, and
the corresponding commit or tag.

| Item | Value |
|---|---|
| Seed(s) | 42 for the main comparison; 42, 1, 2 for the seed study (`scripts/run_a1_seeds.ps1`) |
| Hardware | AMD Ryzen 7 7735HS, 15.2 GB RAM, Windows 11. An RTX 4060 Laptop GPU is present but unused — PyTorch is a CPU-only build, so every reported run is CPU |
| Software | Python 3.10.21, PyTorch 2.14.0+cpu, torchvision 0.29.0, CUDA not used. Exact versions: `requirements.lock.txt` |
| Checkpoint access | Each run writes `results/a1/runs/<run_id>/best.pt` (git-ignored). Reconstruct with the train command below plus that run's `config.yaml` and the committed split |
| Experiment tracking | Run folders under `results/a1/runs/`; no external tracking service |

A pre-run notebook without reproduction instructions is not an acceptable submission.

## Milestones (23:59 GMT+7)

| Due | Assignment | Milestone | Weight |
|---|---|---|---|
| 09 Sep 2026 | Course | Group registration + Pages skeleton | gate |
| 23 Sep 2026 | A1 | M1 Draft | 25% of A1 |
| 07 Oct 2026 | A2 | M1 Dataset proposal | 15% of A2 |
| 21 Oct 2026 | A1 | M2 Final | 75% of A1 |
| 28 Oct 2026 | A2 | M2 Draft | 25% of A2 |
| 11 Nov 2026 | A2 | M3 Final | 60% of A2 |
| 18 Nov 2026 | A3 | M1 Dataset proposal | 15% of A3 |
| 25 Nov 2026 | A3 | M2 Draft | 25% of A3 |
| 02 Dec 2026 | A3 | M3 Final | 60% of A3 |

Late submissions are accepted up to 7 days, at −20% of the milestone score per started calendar week; beyond 7 days
the milestone scores 0 without a prior written exception.

## AI usage

See [AI_USAGE.md](AI_USAGE.md), plus the disclosure sections on the landing page, on each assignment page, and in
each report.
