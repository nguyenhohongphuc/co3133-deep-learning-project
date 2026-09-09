# AI Usage Disclosure — CO3133 Course Project, Group [ID]

This file is the detailed log required by the course handbook. Shorter summaries appear on the landing page, on
each assignment page, and in each report; this file holds the full record.

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

> The group declares that generative AI tools **were used** in this course project. Every use is logged below.

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

### Assignment 2

_(no entries yet)_

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
