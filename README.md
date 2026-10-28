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

---

## Quick start (A1, from a fresh clone)

If you only want the commands, this is the whole pipeline. Every step is explained in the sections below.

```powershell
# 1. Environment (once)
conda create -n co3133 python=3.10 -y
conda activate co3133
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # GPU build; see "Installation"
pip install -r requirements.txt

# 2. Data (once)
python scripts/prepare_data.py --dataset fashion-mnist --out data/

# 3. Train one model  ->  results/a1/runs/<run_id>/
python -m src.train --model linear

# 4. Evaluate that run on the test set  ->  test_metrics.json + figures/
python -m src.evaluate --run_dir results/a1/runs/<run_id>
```

All commands are run from the **repository root** (the folder that contains `src/`), with the
`co3133` environment activated.

---

## Assignment 2 (30VNFoods, image classification)

A2 code lives in its own package `src/a2/` and config `configs/a2.yaml`, so the A1 files are never touched.
It imports the shared `src/engine/trainer.fit` and `src/utils/` without changing them.

```powershell
# 1. Data (once): Kaggle quandang/vietnamese-foods, version 11 -> data/30VNFoods/{Images,Urls,LICENSE}
python -c "import kagglehub; print(kagglehub.dataset_download('quandang/vietnamese-foods/versions/11'))"
#    then copy the printed folder to data/30VNFoods

# 2. EDA + split (already committed; re-run only to reproduce): notebooks/a2_eda.ipynb
#    -> results/a2/splits/split_v2.json, results/a2/tables/a2_drop_list.csv, results/a2/figures/

# 3. Check one batch after preprocessing  ->  results/a2/figures/a2_example_batch.png
python -m src.a2.data

# 4. Train  ->  results/a2/runs/<run_id>/  (config.yaml, history.csv, metrics.json, figures/curves.png)
python -m src.a2.train --model cnn               # baseline
python -m src.a2.train --model resnet50_frozen   # pretrained, backbone frozen
python -m src.a2.train --model resnet50_full     # pretrained, full fine-tuning
```

| Item | A2 |
|---|---|
| Split | `results/a2/splits/split_v2.json`: Train 17,164 / Validate 2,481 / Test 4,984; near-duplicate clusters (pHash) kept in one split; 507 removed images listed with reasons in `results/a2/tables/a2_drop_list.csv` |
| Models (`src/a2/models.py`) | `cnn`: A1 CNN widened to 4 blocks + global average pooling (397 k params); `resnet50_frozen` / `resnet50_full`: torchvision ResNet-50 `IMAGENET1K_V2`, new 30-way head (23.6 M params) |
| Preprocessing (`src/a2/data.py`) | train: `RandomResizedCrop(224)` + flip + `ColorJitter(0.1)`; eval: `Resize(256)` + `CenterCrop(224)`; CNN normalised with Train v2 stats, ResNet with ImageNet stats |
| Checkpoint selection | best validation macro-F1, early stopping patience 5 (same trainer as A1) |
| Proposal | `reports/a2/proposal.md` (revision 2) |

---

## Repository layout

```
configs/
  a1.yaml                 ALL A1 settings: seed, data, training, and one block per model
  a2.yaml                 ALL A2 settings (30VNFoods)
src/
  a2/                     A2 only: data.py (dataset + loaders), models.py (CNN, ResNet-50), train.py
  data/                   split, transforms, Dataset, DataLoader            (TV 1)
  engine/
    trainer.py            training loop, validation, early stopping, checkpointing
    plots.py              curves, confusion matrix, correct/wrong example grids
  models/                 linear, mlp (TV 1) · cnn (TV 2) · rnn, transformer (TV 3)
    __init__.py           MODELS dict: maps a name ("cnn") to its class
  utils/
    seed.py               set_seed, worker_init_fn
    run_dir.py            creates results/a1/runs/<run_id>/
    git_commit.py         current commit hash (+ "-dirty" if src/ or configs/ have changes)
    env.py                Python / PyTorch / CUDA / CPU / GPU info for metrics.json
  train.py                Entry point: python -m src.train --model <name>
  evaluate.py             Entry point: python -m src.evaluate --run_dir <folder>
scripts/                  prepare_data.py, run scripts, make_tables.py, A2 helpers
notebooks/                EDA and error analysis only (never the pipeline)
results/a1|a2/            COMMITTED: splits/, runs/<run_id>/, figures/, tables/
reports/a1|a2/            Report sources, slides, A2 proposal + approval status
docs/                     GitHub Pages site (published from /docs on main)
planning/                 Team task sheet + task guidelines (not a deliverable)
data/                     Datasets - ignored by git, created by scripts/prepare_data.py
AI_USAGE.md               Detailed AI usage log (mandatory)
```

### Who owns what

| Area | Owner | Files |
|---|---|---|
| Data pipeline, Linear + MLP | TV 1 | `src/data/**`, `src/models/linear.py`, `src/models/mlp.py` |
| Training engine, evaluation, plots, CNN | TV 2 | `src/engine/**`, `src/utils/**`, `src/train.py`, `src/evaluate.py`, `src/models/cnn.py` |
| LSTM/GRU, Transformer | TV 3 | `src/models/rnn.py`, `src/models/transformer.py` |

### The two fixed interfaces

These two contracts let the three areas be developed independently. Do not change them without telling the
whole group.

1. **Data:** `build_loaders(config) -> (train_loader, val_loader, test_loader)`. Every batch is
   `(images, labels)` with `images` of shape `[B, 1, 28, 28]` (float, normalized) and `labels` of shape `[B]` (long).
2. **Models:** every model is an `nn.Module` whose `forward(x)` returns **logits** of shape `[B, 10]`.
   No softmax at the end — `nn.CrossEntropyLoss` applies it internally (handbook §11.1).

---

## Installation

### 1. Create the environment

We use conda with Python 3.10 (any Python 3.10 virtual environment also works).

```powershell
conda create -n co3133 python=3.10 -y
conda activate co3133
```

### 2. Install PyTorch

PyTorch must be installed **before** `requirements.txt`, because the right build depends on your machine.

**With an NVIDIA GPU (recommended).** Check your driver with `nvidia-smi`. The top-right corner shows the highest
CUDA version the driver supports. Driver 566.x supports CUDA 12.6:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

For other drivers, pick the matching command on <https://pytorch.org/get-started/locally/>.

**CPU only:**

```powershell
pip install torch torchvision
```

**Check that the GPU is visible:**

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`2.14.1+cu126 True` means the GPU build is installed. If you see `+cpu` or `False`, the code still runs but on
the CPU — see [Troubleshooting](#troubleshooting).

### 3. Install the rest

```powershell
pip install -r requirements.txt
```

The exact versions used for the reported results are frozen in `requirements.lock.txt`
(regenerate it with `pip freeze > requirements.lock.txt` after changing the environment).

---

## Dataset preparation

### 1. Download Fashion-MNIST

```powershell
python scripts/prepare_data.py --dataset fashion-mnist --out data/   # primary dataset
python scripts/prepare_data.py --dataset mnist --out data/           # debugging only (handbook §10)
```

This creates `data/FashionMNIST/raw/`. The `data/` folder is ignored by git, so every member downloads it once.

### 2. The fixed train / validation / test split

**You do not need to run this** — the split is already committed at `results/a1/splits/split_seed42.json`.

| Split | Images | Source |
|---|---|---|
| train | 54,000 | official training set, stratified by class |
| val | 6,000 | official training set, stratified by class |
| test | 10,000 | official test set, untouched |

All models use this same file, which is required for a fair comparison (handbook §12.2).
Rebuild it only if the group agrees to change the split:

```powershell
python -m src.data.split --config configs/a1.yaml
```

---

## Configuration: `configs/a1.yaml`

Every A1 setting lives in this one file. `train.py` reads it and saves a copy into the run folder, so each run
remembers exactly how it was trained.

| Block | Keys | Read by |
|---|---|---|
| top level | `seed`, `data_dir`, `split_file` | `train.py`, `loaders.py` |
| `data` | `val_size`, `stratified` (split) · `normalize`, `mean`, `std`, `augmentation` (transforms) | `split.py`, `transforms.py`, `evaluate.py` |
| `training` | `batch_size`, `num_workers`, `epochs`, `loss`, `optimizer`, `lr`, `weight_decay`, `device`, `monitor`, `patience` | `loaders.py`, `trainer.py` |
| `models.<name>` | constructor arguments of that model | `train.py`, `evaluate.py` |

Things that are easy to get wrong:

- **`models.<name>` keys must match the model's `__init__` arguments exactly.** They are passed as
  `MODELS[name](**config["models"][name])`, so an unknown key raises `TypeError: unexpected keyword argument`.
- **`device`**: `auto` (GPU if available, otherwise CPU), `cpu`, or `cuda`. Never leave it empty.
- **`monitor`** is the validation metric used to pick the best checkpoint (`val_macro_f1`). The trainer keeps
  the epoch with the **highest** value, so the metric must be "higher is better".
- **`patience`**: training stops early after this many epochs without improvement of `monitor`.
- **`num_workers`**: if loading hangs on Windows, set it to `0`.
- For a quick test, set `epochs: 1`, but delete those runs afterwards and set it back before an official run.

---

## Train

```powershell
python -m src.train --model linear
python -m src.train --model mlp
python -m src.train --model cnn
python -m src.train --model rnn
python -m src.train --model transformer
```

`--model` chooses which block of `models:` in `configs/a1.yaml` is used. Everything else comes from the config.

What happens:

1. Seeds are fixed (`seed` in the config).
2. A new folder `results/a1/runs/<run_id>/` is created, with `run_id = a1_<model>_s<seed>_<MMDD-HHMM>`
   (for example `a1_cnn_s42_1023-1430`), and the config is copied into it as `config.yaml`.
3. The model is trained on **train** and validated on **val** after every epoch. The test set is never read.
4. Whenever `monitor` improves, the weights are saved to `best.pt`.
5. At the end, `history.csv` and `metrics.json` are written.

The terminal prints one line per epoch:

```
Epoch 000: train_loss=0.5754, val_loss=0.4384, val_accuracy=0.8487, val_macro_f1=0.8468
```

Epochs are numbered from **0** in the log, in `history.csv`, and in `best_epoch`.

### Output of one training run

| File | Content | In git? |
|---|---|---|
| `config.yaml` | the exact config used for this run | yes |
| `best.pt` | model weights at the best validation epoch | **no** (`*.pt` is ignored, see [Checkpoints](#checkpoints)) |
| `history.csv` | one row per epoch: `train_loss`, `val_loss`, `val_accuracy`, `val_macro_f1` | yes |
| `metrics.json` | `run_id`, `model`, `seed`, `commit`, `device`, `env` (Python/PyTorch/CUDA/CPU/GPU), `param` (number of parameters), `epochs_run`, `train_time` (seconds), `best_epoch`, `best_value`, `monitor` | yes |

---

## Evaluate

```powershell
python -m src.evaluate --run_dir results/a1/runs/a1_cnn_s42_1023-1430
```

Tip: type the start of the folder name and press **Tab** to auto-complete it.

`--run_dir` is required on purpose: the command must name the exact run, so that every reported number can be
traced back to it (handbook §4.2). `evaluate.py` reads `config.yaml` **from the run folder**, not
`configs/a1.yaml`, because the shared config may have changed since that run was trained.

What happens:

1. The model is rebuilt with that run's config, and `best.pt` is loaded.
2. The model predicts the whole **test** set (10,000 images).
3. Metrics are written to `test_metrics.json`, and figures to `figures/`.

> **Run evaluation on the test set once per final model, at the end.** If you look at test results and then go
> back and change hyperparameters, the test set becomes a second validation set and the comparison is no longer fair.

### Output of one evaluation

Added to the same run folder:

| File | Content |
|---|---|
| `test_metrics.json` | `accuracy`, `macro_f1`, `confusion` (10×10, rows = true class, columns = predicted class), `inference_time` (seconds for the whole test set, data loading included), `device`, `model` |
| `figures/curves.png` | training/validation loss and validation accuracy / macro-F1 per epoch |
| `figures/confusion_test.png` | confusion matrix |
| `figures/correct_test.png` | 16 correctly classified test images with their labels |
| `figures/wrong_test.png` | 16 misclassified test images with true and predicted labels |

These cover the mandatory comparison outputs of handbook §12.1: accuracy, macro-F1, number of parameters
(`metrics.json`), training time (`metrics.json`), inference time, curves, confusion matrix, and correct/incorrect
examples.

---

## Recommended workflow for an official run

The `commit` field in `metrics.json` ends with `-dirty` if `src/` or `configs/` had uncommitted changes when the run
started. A dirty run cannot be traced back to a commit, so follow this order:

1. **Smoke test.** Set `epochs: 1`, then train and evaluate once to check that everything works.
   Delete that run folder and set `epochs` back.
2. **Commit the code and config first:**
   ```powershell
   git add -A src configs
   git commit -m "..."
   ```
3. **Train and evaluate:**
   ```powershell
   python -m src.train --model cnn
   python -m src.evaluate --run_dir results/a1/runs/<run_id>
   ```
4. **Check the run:** `commit` has no `-dirty`, and `figures/curves.png` looks sensible.
5. **Commit the results:**
   ```powershell
   git add results/a1/runs/<run_id>
   git commit -m "Add A1 run: cnn seed 42"
   ```

If you change code after step 2, go back to step 2 and train again.

---

## Adding or changing a model

1. Write the class in `src/models/<name>.py`: an `nn.Module` that takes `[B, 1, 28, 28]` and returns logits `[B, 10]`.
2. Register it in `src/models/__init__.py`:
   ```python
   from .rnn import SequenceRNN
   MODELS = {..., "rnn": SequenceRNN}
   ```
3. Put its constructor arguments under `models.<name>` in `configs/a1.yaml`.
4. Add the name to the `choices` list of `--model` in `src/train.py`.
5. Run `python -m src.train --model <name>` with `epochs: 1` to check it.

**GPU note.** Any tensor created inside the model (positional encoding, CLS token, masks…) must be an
`nn.Parameter`, a `register_buffer`, or created with `device=x.device`. Otherwise training on the GPU fails with
`Expected all tensors to be on the same device`.

---

## Checkpoints

`*.pt` is in `.gitignore`, so checkpoints are not committed by default.

- **A1 (included in the repository):** the `best.pt` of all seven reported A1 runs (4.3 MB in total) were
  force-added in commit `44d4fd2` and sit next to the files of their run, at
  `results/a1/runs/<run_id>/best.pt`. Each file is the model `state_dict` only (no optimizer state), selected
  by best validation macro-F1. Evaluate one directly after cloning:
  `python -m src.evaluate --run_dir results/a1/runs/a1_cnn_s42_1027-0157`.
  The runs were trained at commit `c1747a1` (recorded in each `metrics.json`); the A1 model, data and config
  code is unchanged between that commit and `44d4fd2`.
- **Reconstruct:** check out the commit recorded in that run's `metrics.json`, copy the run's `config.yaml` to
  `configs/a1.yaml`, and run `python -m src.train --model <model>`. The committed split and the fixed seed give
  the same data order; small numerical differences on the GPU are possible.

---

## Reproducibility

Every reported number must be traceable to its configuration, dataset split, checkpoint, experiment ID, and commit
(handbook §4.2). Each run folder records all of them:

| Requirement | Where |
|---|---|
| Configuration | `results/a1/runs/<run_id>/config.yaml` |
| Dataset split | `split_file` in that config → `results/a1/splits/split_seed42.json` |
| Checkpoint | `best.pt` (see [Checkpoints](#checkpoints)), selected by `monitor` on the validation set |
| Experiment ID | `run_id` (the folder name) |
| Commit | `commit` in `metrics.json` |
| Seed | `seed` in `metrics.json` and `config.yaml` |
| Hardware and software | `device` and `env` in `metrics.json` |

| Item | Value |
|---|---|
| Seed(s) | 42 for the main comparison |
| Hardware | AMD Ryzen 7 7735HS, 15.2 GB RAM, NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), Windows 11. The device of each run is recorded in its `metrics.json` |
| Software | Python 3.10, PyTorch 2.14.1+cu126, torchvision 0.29.1+cu126, CUDA 12.6. Exact versions: `requirements.lock.txt` |
| Checkpoint selection | highest validation macro-F1, early stopping with patience 5 |
| Experiment tracking | run folders under `results/a1/runs/`; no external tracking service |

All models in the main comparison must be trained and evaluated on the **same device**, since training and inference
times are not comparable between CPU and GPU.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run from the repository root, and use `python -m src.train`, not `python src/train.py`. |
| `ModuleNotFoundError: No module named 'src.common'` | Old code. `src/common/` was replaced by `src/utils/` and the `@register` decorator by the `MODELS` dict; remove the old import. |
| `UnicodeDecodeError: 'charmap' codec can't decode ...` | A file containing non-ASCII text (e.g. Vietnamese comments) was opened without `encoding="utf-8"`. Add it to the `open(...)` call. |
| `torch.cuda.is_available()` is `False` but the machine has an NVIDIA GPU | The CPU build of PyTorch is installed (`torch.__version__` ends with `+cpu`). Reinstall it: `pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu126` |
| `WinError 32 ... being used by another process` during `pip install` | Another program holds a file of the environment (VS Code, a Jupyter kernel, another terminal, antivirus). Close them, then repeat the command. Delete leftover folders starting with `~` in `site-packages`. |
| `WARNING: Ignoring invalid distribution -orch` | Leftover `~...` folders from an interrupted install, in `...\envs\co3133\Lib\site-packages`. Delete only the folders whose names start with `~`. |
| `TypeError: __init__() got an unexpected keyword argument` | A key under `models.<name>` in `configs/a1.yaml` is not an argument of that model. |
| `RuntimeError: Error(s) in loading state_dict ... size mismatch` | The model was rebuilt with different arguments from the ones used in training. Make sure you evaluate with the run's own `config.yaml` (which `evaluate.py` does by default). |
| `NotImplementedError` when training `rnn` or `transformer` | The model is not implemented yet (TV 3). |
| `FileExistsError` when starting a run | Two runs of the same model and seed in the same minute share a `run_id`. Wait a minute and run again. |
| Training hangs at the start on Windows | Set `num_workers: 0` in `configs/a1.yaml`. |

---

## Current status (A1)

| Item | Status |
|---|---|
| Data pipeline, fixed split | done |
| Linear, MLP, CNN | implemented |
| LSTM/GRU, Transformer | not implemented yet (TV 3) |
| `src/train.py`, `src/evaluate.py`, `src/engine/plots.py` | done |
| CNN `dropout` argument | accepted but not used yet: `SmallCNN` contains no `nn.Dropout` layer |
| `scripts/run_a1_all.ps1`, `run_a1_seeds.ps1` | still use the old `--config` interface; to be updated (multi-seed runs also need a `--seed` option in `train.py`) |
| `scripts/make_tables.py` | not implemented yet |

---

## Publishing the site (one-time setup)

1. Push this repository to GitHub (public).
2. **Settings → Pages → Build and deployment**: source *Deploy from a branch*, branch `main`, folder `/docs`.
3. Wait for the first deployment, open the published URL, and paste it into the table above and into the LMS submission.

## Git workflow

- Work on your own branch (e.g. `pnguyen`), never directly on `main`.
- Commit code before official runs (see [Recommended workflow](#recommended-workflow-for-an-official-run)).
- Push your branch and open a Pull Request into `main`. In the description, mention anything that changes the two
  [fixed interfaces](#the-two-fixed-interfaces) or `configs/a1.yaml`.

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
