# AI Usage Disclosure — CO3133 Course Project, Group NNs

This file records AI assistance used in the course project. Related summaries appear on the website and in the reports.

Rules the group follows:

- AI tool use is allowed **only when disclosed and verifiable**.
- Every member is fully responsible for the submitted code, numbers, figures, claims, and narrative, and must be
  able to explain any part of it.
- No AI-generated data, results, citations, or references. No claimed experiment that was not actually run.
- No private, restricted, or credential-bearing data is pasted into AI tools.
- AI-assisted content is verified against source code, real runs, official documentation, or scholarly sources.

Missing or false disclosure may be treated as an academic integrity violation.

---

## Declaration

> The group declares that generative AI tools **were used** in this course project. AI assistance is summarized below.

---

## Log entries

Copy the template below for every distinct use. Group the entries by assignment.

### Course-wide — project website

**Entry 1 — GitHub Pages skeleton**

- **Tool:** Claude (Claude Code, model Opus 5)
- **Used by:** Hồ Hồng Phúc Nguyên (2352824)
- **Time / stage:** 09 Sep 2026 — setting up the course website before the Week-3 "Group + Pages skeleton" milestone
- **Purpose:** coding assistance / document structuring — read `handbook-ene.pdf` and generate the GitHub Pages
  skeleton: the shared landing page and the three assignment page templates, plus `README.md`, `.gitignore`, and
  this file
- **Affected assignment section(s) and files:** `docs/index.html`, `docs/assignments/assignment-1.html`,
  `docs/assignments/assignment-2.html`, `docs/assignments/assignment-3.html`, `docs/assets/css/style.css`,
  `README.md`, `AI_USAGE.md`
- **Representative prompt:** "Đọc file handbook-ene.pdf và làm git page skeleton theo như yêu cầu" (read the
  handbook PDF and build the GitHub Pages skeleton it requires)
- **How the output was edited and verified:** the group compared every generated section against the handbook
  requirements (landing-page fields §2.1, assignment-page minimum content §2.2, report structure §3, deliverables
  §4, disclosure fields §5.2, milestone dates §7, per-assignment rubrics); group information and all links were
  filled in by the group by hand; the pages were opened in a browser and the internal links checked. The tool
  produced templates and placeholders only — **no data, results, figures, or citations were generated.**
- **Sources used for verification:** `handbook-ene.pdf` (course handbook), the LMS course page
- **Member responsible for final verification:** Hồ Hồng Phúc Nguyên (2352824)

### Assignment 1

**Entry 1 — Review of the data pipeline pull request (A1-01, A1-03, A1-04, A1-06, A1-07)**

- **Tool:** Claude (Claude Code, model Opus 5)
- **Used by:** Hồ Hồng Phúc Nguyên (2352824)
- **Time / stage:** 22 Sep 2026 — reviewing the `feature/a1-data-baseline` branch before merging it into `main`
- **Purpose:** coding assistance / debugging — review a teammate's data pipeline and verify its claims
- **Affected assignment section(s) and files:** `src/data/split.py`, `src/data/loaders.py`,
  `src/data/fashion_mnist.py`, `src/data/transforms.py`, `src/models/linear.py`, `src/models/mlp.py`;
  report §3.1 (data splitting, preprocessing)
- **Representative prompt:** "My teammate just pushed some code to a feature branch on GitHub. Could you review
  and merge it for me?"
- **How the output was edited and verified:** the review was run as real checks, not opinion — the split was
  rebuilt from seed 42 and compared index-by-index with the committed `split_seed42.json` (identical), the
  normalization statistics were recomputed on the 54,000 training images only (0.286209 / 0.353160, matching the
  config), and the loaders were driven to confirm the agreed batch contract `[B, 1, 28, 28]` float32 /
  `[B]` int64. The review also found that `notebooks/a1_eda.ipynb` cannot regenerate its own committed figures;
  that finding was passed to TV 1 rather than patched here.
- **Sources used for verification:** PyTorch and torchvision documentation; the committed split file; live runs
- **Member responsible for final verification:** Hồ Hồng Phúc Nguyên (2352824)

**Entry 2 — Training and evaluation pipeline (A1-02, A1-05, A1-11)**

- **Tool:** Claude (Claude Code, model Opus 5)
- **Used by:** Hồ Hồng Phúc Nguyên (2352824)
- **Time / stage:** 22–23 Sep 2026 — implementing the TV 2 training framework before the M1 draft milestone
- **Purpose:** coding assistance / debugging / concept explanation — complete the metrics, trainer, entry points,
  figures and profiling, and explain the checkpoint-selection rule
- **Affected assignment section(s) and files:** `src/engine/metrics.py`, `src/engine/trainer.py`,
  `src/engine/plots.py`, `src/engine/profiler.py`, `src/train.py`, `src/evaluate.py`,
  `configs/a1/base.yaml`; report §3.2 (training setup) and §3.3 (results, figures)
- **Representative prompt:** "write metrics.py and wire up the checkpointing"; earlier in the same session,
  "I thought selecting the checkpoint was based on train and val loss, so why do we need Macro-F1 here?"
- **How the output was edited and verified:** `accuracy`, `macro_f1` and `confusion` were cross-checked against
  `sklearn.metrics` on 5,000 random labels and on hand-computed cases, including the empty-input and
  length-mismatch errors. The loop was verified end to end: `python -m src.train --config configs/a1/linear.yaml`
  trains, checkpoints on validation macro-F1 and early-stops; the same seed run twice produced identical per-epoch
  numbers; `python -m src.evaluate --split test` reproduced test accuracy 0.8356 / macro-F1 0.8354 for the linear
  model and generated the §12.1 figures, which were inspected by eye. Several defects in the AI-written or
  AI-suggested code were found this way and fixed: a missing `pandas` import that killed `fit()` *after* training,
  three missing imports in `plots.py`, a `ZeroDivisionError` in `inference_time` on short loaders, and a
  `UnicodeEncodeError` from printing Vietnamese to the cp1252 Windows console.
- **Later revision (23–27 Oct 2026):** Nguyên removed `metrics.py` and `profiler.py` and rewrote `src/train.py`
  and `src/evaluate.py` by hand (commits `0e2d047`, `e9b49af`, `6984a3e`). `src/engine/trainer.py` and
  `src/engine/plots.py` remain largely the AI-assisted September version, edited by hand.
- **Sources used for verification:** scikit-learn and PyTorch documentation; the course handbook §§3.2, 3.3, 4.2,
  12.1, 12.2; live training and evaluation runs committed as `results/a1/runs/<run_id>/`
- **Member responsible for final verification:** Hồ Hồng Phúc Nguyên (2352824)

**Entry 3 — Documentation of the A1 setup**

- **Tool:** Claude (Claude Code, model Opus 5)
- **Used by:** Hồ Hồng Phúc Nguyên (2352824)
- **Time / stage:** 23 Sep 2026 — filling the assignment page and README before the M1 draft
- **Purpose:** report writing / document structuring — write up the dataset, EDA, split, preprocessing, training
  setup and environment sections from the values the code actually uses
- **Affected assignment section(s) and files:** `docs/assignments/assignment-1.html`, `README.md`,
  `requirements.lock.txt`, this file
- **Representative prompt:** "update docs for me"
- **How the output was edited and verified:** every number written into the page was taken from a real artefact —
  the committed split file, the config, `pip freeze`, and the machine's own CPU/RAM/GPU report — not from memory.
  Sections with no results yet (CNN, LSTM/GRU, Transformer, the comparison table, slides, video) were left marked
  TODO rather than filled with plausible text.
- **Sources used for verification:** the repository's own files and run outputs; course handbook §§2.2, 3, 4.2
- **Member responsible for final verification:** Hồ Hồng Phúc Nguyên (2352824)

**Entry 4 — Final A1 assignment page and landing-page roles**

- **Tool:** Claude (Claude Code, model Opus 5.5)
- **Used by:** Hồ Hồng Phúc Nguyên (2352824)
- **Time / stage:** 28 Oct 2026 — preparing the M2 Final package
- **Purpose:** report writing / result analysis — rewrite the A1 page with the final seven-run comparison, and
  fill each member's role on the landing page from the git history
- **Affected assignment section(s) and files:** `docs/assignments/assignment-1.html`, `docs/index.html`,
  `docs/assets/img/a1/` (figures copied from the report branch), this file
- **Representative prompt:** "cập nhật assignment-1.html và vai trò landing page đi"
- **How the output was edited and verified:** every number on the page was recomputed from the committed run
  folders (`metrics.json`, `test_metrics.json`, `history.csv`) and `results/a1/tables/comparison.csv`, and checked
  against the report tables and slide 21; parameter counts were checked by hand from the layer sizes. Figures are
  the report's own, not regenerated. [member to add: final read-through result]
- **Sources used for verification:** committed run outputs; `src/models/*.py`; course handbook §§2.2, 12.1, 12.2
- **Member responsible for final verification:** Hồ Hồng Phúc Nguyên (2352824)

<!-- template — copy per use -->
- **Tool:** [tool name + version/model if known]
- **Used by:** [member name]
- **Time / stage:** [date; e.g. implementation of the training loop]
- **Purpose:** [category: literature search / concept explanation / architecture or experiment design /
  coding assistance / debugging / test generation / report writing / grammar check / figure or slide generation /
  result analysis]
- **Affected assignment section(s) and files:** [e.g. `src/train.py`; report §3.2]
- **Representative prompt (or prompt-log link):** [short summary or link]
- **How the output was edited and verified:** [what was changed; what was tested]
- **Sources used for verification:** [docs / papers / tests]
- **Member responsible for final verification:** [member name]

**A1 report — collaborative writing with Nguyễn Nhật Nam**

- **Tool / user / dates:** OpenAI Codex (model identifier not recorded); Nguyễn Nhật Nam (2352778); 27–28 Oct 2026.
- **Purpose and workflow:** The student and AI developed the report section by section. The student directed the scope, section order, revisions, and review process. AI assisted with drafting, explanations, analysis of existing results, figures, and LaTeX formatting according to those instructions. Completion of the remaining sections for one consolidated review was explicitly requested by the student.
- **Affected files:** `reports/a1/`, related report-generation scripts, and supporting analysis files in `results/a1/tables/`.
- **Verification and responsibility:** Drafts and figures were checked against repository code, saved experiment outputs, source images, and cited references; the PDF was compiled and visually inspected. No new training or checkpoint inference was performed. Final human review is pending; Nguyễn Nhật Nam and the group remain responsible for the submitted report.

**Entry 5 — LSTM and Transformer implementation assistance**

- **Tool:** ChatGPT (OpenAI GPT; exact model version not recorded)
- **Used by:** Phan Quốc Đại Sơn (2353053)
- **Time / stage:** October 2026 — implementation, local testing, and debugging of the Assignment 1 sequence models
- **Purpose:** coding assistance / debugging / concept explanation — support the implementation of
  image-to-sequence conversion for rows, columns, and patches; develop the LSTM and Patch Transformer classifiers;
  explain tensor shapes, LSTM hidden states, patch embeddings, CLS tokens, positional embeddings, and Transformer
  encoders; and troubleshoot training and evaluation issues
- **Affected assignment section(s) and files:** `src/models/rnn.py`, `src/models/transformer.py`, related A1
  configurations and training/evaluation commands; report Sections 5.4–5.5 and the LSTM representation comparison
- **Representative prompts:**
  - "Implement SequenceRNN with row, column, and patch inputs and explain how the image becomes a sequence."
  - "Implement the Patch Transformer with a CLS token and learned positional embeddings."
  - "Explain and fix the checkpoint/evaluation error."
- **How the output was edited and verified:** AI-generated suggestions were reviewed and adapted to the project's
  existing PyTorch model interface and configuration. Model training and evaluation commands were executed locally
  during development. Tensor shapes, model outputs, and evaluation metrics were checked against actual experiment
  results. Final reported metrics were obtained from recorded experiments rather than generated by ChatGPT.
- **Sources used for verification:** project source code, model configuration files, local training and evaluation
  outputs, and official PyTorch LSTM and Transformer documentation
- **Member responsible for final verification:** Phan Quốc Đại Sơn (2353053)

### Assignment 2

- **Tool:** Claude (Claude Code, model Opus 5.5)
- **Used by:** Hồ Hồng Phúc Nguyên (2352824)
- **Time / stage:** 07/10/2026; A2-M1 Dataset Proposal
- **Purpose:** concept explanation (handbook requirements for the proposal), result analysis, report writing
- **Affected assignment section(s) and files:** `reports/a2/proposal.md` (all sections), `reports/a2/approval.md`;
  ran `scripts/a2_eda.py` and `scripts/a2_benchmark.py` to produce `results/a2/eda/` and `results/a2/compute/`
- **Representative prompt (or prompt-log link):** "Read the handbook carefully and tell me what the proposal will contain
  (image classification, Kaggle quandang/vietnamese-foods); run the two EDA files and write the proposal in English"
- **How the output was edited and verified:** every number in the proposal was copied from the generated
  `summary.json`/CSV/JSON files; near-duplicate pairs were inspected visually (all distance-0 examples and 8 random
  distance-4 pairs); the licence (MIT) was checked by the member on the Kaggle page; GPU timings were re-checked with a
  synthetic batch. The paper title was not confirmed, so only the IEEE link is cited.
- **Sources used for verification:** course handbook §§3, 7.5, 17–23; Kaggle dataset page; IEEE Xplore document 9530774;
  the repository's own EDA and benchmark outputs
- **Member responsible for final verification:** Hồ Hồng Phúc Nguyên (2352824)

- **Tool:** Claude (Claude Code, model Opus 5.5)
- **Used by:** Hồ Hồng Phúc Nguyên (2352824)
- **Time / stage:** 27–28/10/2026; EDA redo, proposal revision 2, A2-M2 Draft
- **Purpose:** concept explanation (handbook §§7.5, 18–21), result analysis, coding assistance, report writing
- **Affected assignment section(s) and files:** review of `notebooks/a2_eda.ipynb` (the EDA itself was written and run
  by the member on Google Colab, then re-run locally); `reports/a2/proposal.md` (revision 2); `configs/a2.yaml`;
  `src/a2/data.py`, `src/a2/models.py`, `src/a2/train.py`; `docs/assignments/assignment-2.html`; `README.md`
- **Representative prompt (or prompt-log link):** "check my EDA for remaining problems"; "update the data proposal to
  match the handbook and what we did"; "write the A2 training code and fill in the Pages requirements"
- **How the output was edited and verified:** `split_v2.json` was checked against every local image (path exists, MD5
  matches, label matches the folder, no image in two splits, counts match the drop list). The data loader was checked
  on one batch (`results/a2/figures/a2_example_batch.png`: shape [16, 3, 224, 224], labels 0–29), and both models
  with a forward pass and their parameter counts. Every number on the page and in the proposal comes from the notebook
  outputs, `split_v2.json`, the 07/10 benchmark JSONs, or the `metrics.json`/`history.csv` of a run in
  `results/a2/runs/`. Sections without results yet (controlled experiment, error analysis, test metrics) are marked TODO.
- **Sources used for verification:** course handbook; torchvision documentation (ResNet-50 weights, transforms);
  the repository's own run outputs
- **Member responsible for final verification:** Hồ Hồng Phúc Nguyên (2352824)

### Assignment 3

_(no entries yet)_

---

## Worked example (format reference only — delete before submission)

- **Tool:** [tool/model name]
- **Used by:** [member name]
- **Task:** Debugging the validation loop in `train.py`
- **Prompt summary:** Asked why validation loss was accumulated incorrectly
- **AI contribution:** Suggested sample-weighted loss accumulation
- **Student verification:** Compared with the PyTorch documentation; tested on a small controlled set
- **Affected files/sections:** `src/train.py`; report §3.2
- **Responsible member:** [member name]
