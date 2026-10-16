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

---

## Phase 3 — Defining "falling behind"

### Concept
A **target** (label) is the thing a supervised model learns to predict. Its definition decides
what the model can ever mean. **Target leakage** happens when a feature contains information
that would not exist at prediction time — the model then looks brilliant in testing and fails
in use.

### Why it was required
Your file has no outcome column, so "at risk" had to be *defined*. A vague definition ("weak
student") cannot be evaluated; a concrete one ("fails a subject or SGPA < 6 next semester") can.

### What we implemented
`targets.py` turns marks into pass/fail, grade points and SGPA, flags difficulty semesters, and
attaches to each semester *t* the outcome of semester *t + 1*. Unknown stays unknown (NaN).

### Small example
A student with SGPA 7.4 in semester 4 who fails one subject in semester 5 → the semester-4 row
gets label 1. Their semester-5 row has no semester 6 yet → `future_semester_not_observed`; the
app will *score* it, but it is never used for training.

### Common mistakes
- Using semester *t*'s ETE to predict semester *t*'s result (leakage).
- Treating "no record" as "no problem": students who leave are often the ones struggling.
- Presenting illustrative thresholds as official rules.

### How to explain it to a professor
"The label is computed by a fixed rule from the *next* semester's marks. The model only sees
information available at the end of the current semester, and rows whose future is unknown are
excluded from training rather than assumed safe."

---

## Phase 4 — Building a synthetic dataset

### Concept
A **generative model** is a recipe that produces data from hidden causes. Writing the recipe
ourselves means every relationship in the data is known, documented and reproducible — but also
that a model trained on it can only learn *our* recipe.

### Why it was required
No public dataset matches the college format, and the reference file has one semester only.
The LSTM needs genuine semester-by-semester histories; this generator provides them.

### What we implemented
Hidden traits (ability, engagement, trajectory) → semester states with carry-over and shocks →
attendance and marks per subject → outcomes computed with the project's own target code →
attrition → missing values → export in the exact reference layout. Parameters were tuned on a
separate development seed so the official dataset never influenced its own design.

### Small example — why the label is not trivial
Among synthetic students whose current SGPA is between 6.0 and 6.5, about 30 % have a difficulty
semester next — the rest do not. Same visible history, different futures, because next
semester's exam noise and shocks are not visible today. That is what makes prediction a
*probability*, not a lookup.

### Common mistakes
- Generating the label directly from the same features the model sees (the model then learns
  the formula, with perfect but meaningless scores).
- Tuning the generator on the same data you will report results on.
- Copying a reference file's identities, or its errors, into "synthetic" data.
- Forgetting that synthetic metrics measure agreement with the recipe, not real-world accuracy.

### How to explain it to a professor
"I wrote a documented generative model for multi-semester records in our college's format. It
produces marks, not labels; the label is computed afterwards from the next semester's marks,
which contain randomness the model cannot see. The data is reproducible to the byte, passes the
same validator as uploaded files, and every result on it is explicitly a synthetic-data result."
