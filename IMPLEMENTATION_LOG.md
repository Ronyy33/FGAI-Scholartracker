# Implementation Log

Chronological record of what was done, the commands run, problems met and how they were solved.
Newest entries at the bottom.

---

## 2026-10-16 — Milestone 1: inspection, research, scaffold

### Workspace inspection
- `ls -la`, `git status`, `git log` → repository initialised but **empty** (no commits, no files),
  on branch `claude/modest-bell-box1op`; remote `origin` = `github.com/Ronyy33/FGAI-Scholartracker`.
- Remote has no branches yet (GitHub API `list_branches` → `[]`). Repository visibility: **public**.
- Tools: Python 3.13.16, uv 0.11.32, Node 22.22.0, npm 10.9.4, pnpm, yarn, Docker, git.
  Pre-installed Python packages include numpy 2.5.3, pandas 3.0.5, pydantic 2.13.5, uvicorn 0.53.0;
  scikit-learn, PyTorch, FastAPI and pytest were **not** installed.
- Hardware: 4 CPUs, 15 GiB RAM, no GPU.

### Reference CSV search
- `find / -iname "*scholartrack*"` and a CSV search under home directories → not found.
- GitHub remote → no branches/files. Google Drive search (title and full-text for `scholartrack`,
  `PRN`, `synthetic_200`, `credit_weighted_marks_pct`) → no results.
- **Outcome:** file not available. Per the working rules, no substitute file is used; the file
  has been requested.

### Network constraints discovered
- `curl` through the environment proxy: `archive.ics.uci.edu` and `analyse.kmi.open.ac.uk` →
  `CONNECT tunnel failed, response 403`. Also unreachable: openml.org, zenodo.org, kaggle.com,
  huggingface.co, nature.com, mdpi.com, arxiv.org, `download.pytorch.org`.
- Reachable: pypi.org, files.pythonhosted.org, registry.npmjs.org, raw.githubusercontent.com,
  github.com.
- **Consequence:** dataset facts could only be confirmed through web-search summaries and one
  GitHub mirror README; each fact in `research/DATASET_RESEARCH.md` is tagged with a
  verification level. No dataset was downloaded.
- **Consequence for later phases:** PyTorch CPU wheels cannot be fetched from
  `download.pytorch.org` here; PyPI's Linux wheel pulls CUDA libraries. To be handled in Phase 6.

### Dataset research
- Web searches covering UCI (ids 697, 320, 467, 856), OULAD, Kaggle student-risk datasets,
  Indian engineering-college datasets (PredictGrad), plus newer leads.
- Read directly: PredictGrad repository page; OULAD GitHub mirror README.
- Written: `research/DATASET_RESEARCH.md`, `research/DATASET_COMPARISON.md`, `data/README.md`.
- Conclusion: no public dataset matches the college schema with an observed future outcome.
  Recommended two-track plan (synthetic college format + UCI 697 real-outcome benchmark).

### Scaffold
- Created `docs/PROJECT_PLAN.md`, `README.md`, `TASKS.md`, `PROJECT_STATUS.md`, this log,
  `docs/LEARNING_NOTES.md`, `.gitignore`, `pyproject.toml`, `src/scholartrack/__init__.py`,
  `tests/test_repo_hygiene.py`, `data/{reference,external,synthetic,processed}/.gitkeep`.
- `uv sync` → **failed** first time: hatchling requires `README.md`, which had not been written
  yet. Fixed by writing the README; second `uv sync` installed pytest 9.1.1, ruff 0.17.1.
- `uv run pytest -v` → **11 passed**.
- `uv run ruff check .` → clean. `uv run ruff format --check .` → 1 file needed formatting;
  ran `ruff format`, re-check clean.
- Negative check: planting `docs/leak.csv` and `src/.env.local` did **not** fail the tests,
  because `.gitignore` already makes them uncommittable (intended). Force-adding the CSV with
  `git add -f` **did** fail `test_no_data_files_outside_fixtures`, showing the guard works;
  the file was then removed and the suite returned to 11 passed.
- uv warned that `UV_NATIVE_TLS` is deprecated in favour of `UV_SYSTEM_CERTS` (environment
  setting, harmless).

### Git
- Committed locally on `claude/modest-bell-box1op`. **Not pushed** — waiting for your explicit
  approval as required by the project's working rules.

---

## 2026-10-16 — Git: making the work visible on GitHub

- You uploaded the reference CSV to `main` (commit `fa4f9f7`, "Add files via upload").
- My branch had been created from an empty repository, so it shared no history with `main`.
  Rebased it onto `origin/main` (safe: the branch had never been pushed). Conflict in
  `README.md` (your one-line placeholder vs. the project README) resolved by keeping the
  project README.
- The rebased branch now contained the CSV at the repository root, and
  `test_no_data_files_outside_fixtures` **failed** — the guard worked as designed.
- Copied the CSV byte-for-byte to `data/reference/` (Git blob `4879a4a…` identical, SHA-256
  `2746d846…`), made the copy read-only, then removed the root copy from this branch in its own
  commit ("Stop tracking the reference CSV on this branch"). `main` is unchanged.
- Pushed `claude/modest-bell-box1op`; verified with `git ls-remote`.

## 2026-10-16 — Phase 2: reference CSV inspection and validation

### Findings (exploration, then confirmed by code)
- 200 rows × 66 columns, one semester (5, 2026-27), UTF-8 **with BOM**. No outcome column.
- 12 impossible values in 12 different rows (SGPA 12.4, attendance 145 % and 132 %, CIE −5 and
  55/40, ETE 78/60, termwork −3/25 and 35/25, study hours −4, commute −15, gender "Unknown??",
  part-time job "Sometimes"), 1 unknown elective, 2 stray columns, 20 scattered blank cells.
- Credits are not in the file but were **recovered exactly** by least squares from
  `credit_weighted_marks_pct` (and independently from `overall_attendance_pct`):
  3,3,3,3,3,1,1,2,1,2,1 = 23 credits; residuals ≤ 0.06 (rounding).
- In all five component errors the subject *total* is correct; the component was altered.
- Internal inconsistency: prior SGPA is tightly clustered (sd 1.11) but semester-5 marks imply
  SGPA sd 2.65; 66 % of students would fail ≥ 1 subject under an illustrative 40 % rule, 84 % of
  those with prior SGPA 6.5–7.5. Verified independently from the raw file.

### Implemented
- `configs/curriculum.yaml` (semester 5 inferred; semesters 1–4, 6–8 illustrative; grading rules
  illustrative), `configs/target.yaml`.
- `src/scholartrack/data/`: `schema.py`, `curriculum.py`, `loader.py`, `validation.py`,
  `adapter.py`, `profiling.py`; `src/scholartrack/targets.py`, `reporting.py`, `paths.py`.
- `scripts/inspect_reference_data.py` → `reports/data_quality_report.md` (deterministic, no
  identifiers).

### Problems met and fixed
- pandas silently renames duplicate headers (`a` → `a.1`), so a duplicate-column check after
  `read_csv` can never fire. The loader now inspects the header line itself.
- pandas 3 rejects positional `axis` (`df.any(1)`) — used `axis=1`.
- A test expected one warning when a total was altered, but the change also (correctly) moved
  `credit_weighted_marks_pct` beyond its tolerance; the test was made stricter to expect both.
- A test helper cast a text column to float; fixed the helper, not the module.
- Reading the generated report revealed four hand-written sentences that disagreed with the
  computed numbers (correlations quoted from raw rather than cleaned data; "each blank in a
  different column" was false; a hard-coded "yes"; an unmeasured claim). All four are now
  computed. Lesson: never hand-write numbers into a generated report.

### Checks
- `uv run pytest` → **84 passed** (reference-file tests run locally; they skip in CI where the
  file is absent). `ruff check` and `ruff format --check` clean. Report regenerated twice →
  byte-identical; contains no PRN and no surname from the file (tested).

## 2026-10-16 — Phase 3: target definition

- Wrote `docs/TARGET_DEFINITION.md`: end of semester *t* → probability that semester *t + 1* is
  an academic-difficulty semester (≥ 1 failed subject OR SGPA < 6.0). Thresholds illustrative and
  configurable; the 10-point letter scale cited (Vikaspedia CBCS page), with the note that
  percentage cut-offs differ between universities and one hosted copy of the UGC guidelines
  shows a different scale. A first search returned only a software vendor's page, which was not
  used as a source.
- `targets.py` uses three-valued logic (1 / 0 / unknown) so a missing ETE never silently becomes
  "not at risk"; `label_status` records why a row is unlabelled.

## 2026-10-16 — Phase 4: synthetic data generator

- `configs/synthetic.yaml`, `src/scholartrack/synthetic/generator.py`,
  `scripts/generate_synthetic_data.py`, `tests/test_synthetic.py`,
  `reports/synthetic_data_design.md`, generated `reports/synthetic_data_profile.md`.
- Calibrated on development seed 1 (five configurations; table in the design report), then
  generated the official dataset with seed 20261016: 1,200 students, 8,561 student-semesters,
  88,452 subject results, 44 college-format files; label rate 20.5 % among 7,300 labelled rows.
- **Problems met and fixed**
  - My first draft of `generate()` ended with placeholder code and never collected the subject
    tables; caught on review before running, rewritten.
  - The profile initially contained three statements not supported by its own numbers
    (attrition "after poor semesters" was not shown by the table; "every band contains both
    outcomes" was asserted; prior SGPA called "close" despite a wider spread). All three are now
    computed; the wider spread was traced to failed subjects scoring 0 grade points (sd 0.99
    among students without a failure vs 1.42 overall).
  - pandas 3 turns a nullable-integer column back into floats when converted to a NumPy array
    if it contains a blank, so some marks printed as `22.0`. The reference file has the same
    quirk; fixed in our export anyway, with a test.
  - The calibration table in the design report first listed three rounds; five configurations
    were actually tried. Corrected.
- **Checks:** 12 synthetic-data tests pass, including CSV → loader → validator → adapter
  round-trip equal to the generator's tables, zero validation issues across all 44 exports,
  semester-5 export column-identical to the reference (minus its two stray columns), and
  byte-identical regeneration (all 48 file hashes and the profile).

## 2026-10-16 — Phase 5: features, splits, preprocessing, data dictionary

- `configs/features.yaml`; `src/scholartrack/features/` (`specs.py`, `build.py`, `split.py`,
  `preprocess.py`); `scripts/build_data_dictionary.py` → `docs/DATA_DICTIONARY.md`;
  `tests/test_features.py`.
- **Design:** 15 *snapshot* features computable from one college-format row (semester t +
  `prior_sgpa`) feed logistic regression and the MLP; the LSTM receives the same vector for each
  semester 1…t. Excluded by default with written reasons: identifiers, gender, age, part-time job,
  commute, elective, division, academic year; the reported summary columns are recomputed from
  subject data instead of trusted.
- **Split:** by student, 70/15/15 within each admission cohort (seed 20261017). On the official
  synthetic data: 5,112 / 1,088 / 1,100 labelled rows; positive rates 20.7 % / 19.3 % / 20.4 %.
- **Preprocessor:** missing indicators + training-median imputation + training standardisation,
  stored as JSON (readable, no pickle).
- **Problem met:** a check that recomputed attendance equals the reported overall attendance held
  for only 94.4 % of rows. Cause: the generator computes the reported column before blanking
  cells, so rows with a blank subject cell are recomputed from fewer subjects. The test now
  restricts to rows with complete subject data and requires a 100 % match (stricter, passes).
- **Checks:** 14 feature tests, including no-look-ahead (features and sequences for semester t
  are identical after deleting all later semesters), train/serve consistency (features from the
  exported CSV after adapting == features from the generator's tables), and a test that the data
  dictionary matches the code.

## 2026-10-16 — Phase 6: models and training

- Network check: `archive.ics.uci.edu`, `download.pytorch.org`, `analyse.kmi.open.ac.uk` still
  blocked. Installed scikit-learn 1.9.1 and PyTorch 2.14.1 from PyPI (`uv add`, 26 s); the Linux
  PyPI build bundles GPU libraries (`.venv` now 5.5 GB) but runs on CPU (CUDA unavailable).
- Code: `src/scholartrack/models/` (networks, baselines, artefacts), `src/scholartrack/training/`
  (data, loop, experiment), `src/scholartrack/evaluation/metrics.py`, `configs/experiment.yaml`,
  `scripts/train.py`, `scripts/evaluate.py`, `scripts/diagnose_signal.py`, `tests/test_models.py`.
- Official run `20261016T200932Z-8acaad7` (52 s on 4 CPU threads). Validation: logistic
  regression log loss 0.2615 / ROC-AUC 0.926 / PR-AUC 0.787; MLP 0.2636 / 0.923 / 0.784;
  LSTM 0.2641 / 0.923 / 0.777; majority 0.4912 / 0.5 / 0.193. Details in `docs/EXPERIMENT_LOG.md`.
- Verified: deterministic retraining (identical metrics), reload reproduces predictions
  (≤ 1.1e-16), test partition untouched (guard + a test that removes the test arrays).
- Exploratory diagnostic (then made reproducible as `scripts/diagnose_signal.py`): history
  summaries add nothing; an oracle on hidden variables reaches only ROC-AUC 0.936.
- Problems fixed: `float(loss)` warning → `loss.item()`; report helper re-parsed formatted text
  (`2.5000`) and printed float dust (`28.0` vs `17`) → `fmt` now keeps text and rounds first
  (data-quality report regenerated: only that table's formatting changed). My own docstring for
  the diagnostic claimed "18 deployed inputs" when it used 15 features — corrected.
- Tests: 120 passing.
