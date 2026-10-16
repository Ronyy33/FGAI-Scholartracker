# Learning Notes

One section per phase: the concept, why it was needed, what we implemented, a small example,
common mistakes, and how to explain it to a professor.

---

## Phase 0 — Project setup and working rules

### Concept
A machine-learning project is software first. Before any model, we need: a known starting
state, a plan with checkable milestones, rules about data, and automated checks that stop
mistakes from creeping in.

### Why it was required
- **Reproducibility:** someone else (your examiner) must be able to rebuild every result.
- **Privacy:** the GitHub repository is *public*. One careless `git add .` could publish a
  student-level CSV, and deleting it later does not remove it from Git history.
- **Honesty:** tracking files (`TASKS.md`, `PROJECT_STATUS.md`, `IMPLEMENTATION_LOG.md`) make it
  obvious what is done versus planned.

### What we implemented
- A `.gitignore` that ignores everything in `data/` (except its README) and every tabular /
  model file type everywhere except `tests/fixtures/`.
- `tests/test_repo_hygiene.py`, which asks Git for every file it *would* commit and fails if any
  data, model or `.env` file is among them.
- A Python project managed by **uv**, which writes a lockfile (`uv.lock`) pinning exact
  versions so installs are identical on every machine.

### Small example
```bash
$ git add -f docs/leak.csv      # force-add a CSV, bypassing .gitignore
$ uv run pytest -q
FAILED tests/test_repo_hygiene.py::test_no_data_files_outside_fixtures
AssertionError: Data/model files would be committed: ['docs/leak.csv']
```
We ran exactly this to prove the test can fail, then removed the file.

### Common mistakes
- Writing a test that can never fail (always check the negative case).
- Committing data "just for now" — Git history keeps it forever.
- Installing packages without a lockfile, so the project breaks months later.

### How to explain it to a professor
"Before writing ML code, I set up guard-rails: data never enters the public repository — that's
enforced by `.gitignore` and by an automated test I verified can catch a violation — and every
dependency is pinned in a lockfile so results are reproducible."

---

## Phase 1 — Dataset research

### Concept
A model can only learn what its **label** contains. A "risk" model trained on a label that
someone invented measures agreement with that invention, not real-world risk. So the first
question is not "which model?" but "**which observed outcome** can we learn from, and are we
allowed to use it?"

### Why it was required
We needed data with: (1) real outcomes defining "falling behind", (2) a licence permitting
reuse, (3) features available *before* the outcome, and (4) for the LSTM, several time steps
per student.

### What we did
- Searched UCI, OULAD, Kaggle, GitHub and India-specific sources.
- Recorded for each: source, size, unit of observation, features, outcome, licence, sensitive
  fields, leakage risks, suitability for MLP/LSTM.
- Tagged every fact with **how** it was verified (fetched page, official-page summary,
  secondary source), because the environment blocks some official sites.

### Small example — why "unit of observation" matters
- UCI 697: one row = one student (enrolment + semester 1 + semester 2) → 2 time steps.
- OULAD: one student has hundreds of dated activity records → real sequences.
An LSTM needs the second kind. Feeding it one flat row per student and calling it "temporal"
would be misleading.

### Common mistakes
- Assuming a Kaggle dataset is real because it is popular — many are generated, with no
  documented origin.
- Trusting mirror sites for licences (we found mirrors listing CC0, CC BY-SA and "unknown" for a
  dataset that UCI lists as CC BY 4.0).
- Ignoring *when* a feature was recorded. If dropout is the outcome, features recorded after
  the student left reveal the answer (**target leakage**).

### How to explain it to a professor
"No public dataset matched our college's format and also had an observed future outcome. So I
use a documented synthetic dataset in our college format to demonstrate the full system,
including genuine semester sequences for the LSTM, and — pending approval — a real public dataset
with observed dropout/graduation outcomes as a benchmark, so at least one set of results
reflects real students."

---

## Phase 2 — Inspecting and validating the reference CSV

### Concept
**Data validation** checks every value against what is *possible* before anything is learned
from it. A **schema adapter** converts a source's own layout into one internal format so the rest
of the system does not care where data came from.

### Why it was required
Your reference file contains 12 impossible values (SGPA 12.4, attendance 145 %, negative marks…).
A neural network would happily learn from them. Removing just those 12 cells raised the
correlation between prior SGPA and marks from 0.74 to 0.83 — bad values hide real patterns.

### What we implemented
- `validation.py` returns structured issues with three severities: **error** (row unusable),
  **invalid** (value impossible → treated as missing), **warning** (suspicious, kept).
- `adapter.py` turns the wide file (one row per student, 5 columns per subject) into two tidy
  tables — one row per student-semester, one row per student-semester-subject — and replaces
  the PRN with an opaque key.
- `profiling.py` recovered the **credits** (not in the file!) by least squares.

### Small example — recovering hidden credits
`credit_weighted_marks_pct = Σ creditᵢ × percentᵢ / Σ creditᵢ`. With 200 students and 11
unknown weights, we solve 200 equations by least squares. The weights came out as
0.130 / 0.087 / 0.043 — exactly 3/23, 2/23, 1/23 — and the rounded credits reproduce the column
to within 0.05 points, i.e. rounding error. Attendance gave the same credits independently.

### Common mistakes
- "Fixing" an impossible value by guessing (e.g. clipping 145 % to 100 %). We mark it missing
  instead; a guessed value is fabricated data.
- Treating a structural zero as a score: the open-elective's CIE is 0 for *everyone* because the
  subject has no CIE, not because every student scored 0.
- Trusting library defaults: pandas silently renamed a duplicate column (`a` → `a.1`), which
  would have hidden a broken file.
- Hand-typing numbers into a report. Four of my hand-written sentences turned out wrong once I
  compared them with the computed values; now the report computes every number it states.

### How to explain it to a professor
"Before modelling, I validated every value against its permitted range and the subject's
assessment scheme. Impossible values become missing rather than being guessed. I also found that
the credit-weighted columns let me recover the credit of every subject exactly, which the rest
of the pipeline needs to compute SGPA."
