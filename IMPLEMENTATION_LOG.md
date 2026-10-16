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
