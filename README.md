# CO3133 — Deep Learning and Its Applications — Course Project (Semester 261)

Group **[ID]** · Faculty of Computer Science and Engineering · Ho Chi Minh City University of Technology (HCMUT), VNU-HCM
Instructor: Lê Thành Sách · LMS: <https://lms.hcmut.edu.vn/course/view.php?id=142848>

Course website (GitHub Pages): **<https://nguyenhohongphuc.github.io/-co3133-deep-learning-project/>**
Repository: **<https://github.com/nguyenhohongphuc/-co3133-deep-learning-project>**

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
docs/                     GitHub Pages site (published from the /docs folder on main)
  index.html              Shared landing page
  assignments/            One page per assignment
  assets/css/style.css    Site stylesheet
configs/                  Experiment configuration files (one per reported run)
src/                      Source code (data, models, training, evaluation)
scripts/                  Entry-point scripts / helper commands
AI_USAGE.md               Detailed AI usage log (mandatory)
```

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
# TODO: exact commands, download source, expected folder layout, checksum
python scripts/prepare_data.py --dataset fashion-mnist --out data/
```

## Train

```bash
# TODO
python src/train.py --config configs/a1_linear.yaml
```

## Evaluate

```bash
# TODO
python src/evaluate.py --config configs/a1_linear.yaml --checkpoint checkpoints/a1_linear_best.pt
```

## Reproducibility

Every reported number must be traceable to its configuration, dataset split, checkpoint, log / experiment ID, and
the corresponding commit or tag.

| Item | Value |
|---|---|
| Seed(s) | [seed] |
| Hardware | [GPU / CPU / RAM] |
| Software | Python [x], PyTorch [x], CUDA [x] |
| Checkpoint access | [link or reconstruction instructions] |
| Experiment tracking | [tool / log location] |

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
