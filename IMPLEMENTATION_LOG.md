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
