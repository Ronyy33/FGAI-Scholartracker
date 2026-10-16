# Data

> **The GitHub repository is public.** No student-level file, real or synthetic, is ever
> committed. Everything under `data/` except this README and the `.gitkeep` placeholders is
> ignored by Git, and `tests/test_repo_hygiene.py` fails if a data file becomes committable.

## Layout

| Folder | Contents | In Git? | How it gets there |
|--------|----------|---------|-------------------|
| `data/reference/` | Your synthetic reference CSV `scholartrack_synthetic_200_students_PRN_sorted.csv` | No | You place it here manually. **Read-only: never modified in place.** |
| `data/external/` | Public datasets (e.g. UCI 697), exactly as downloaded | No | Download script (Phase 4) that records URL, date and SHA-256 |
| `data/synthetic/` | Generated college-format synthetic dataset | No | `scripts/generate_synthetic_data.py` (Phase 4), fixed seed |
| `data/processed/` | Cleaned / feature-engineered tables | No | Pipeline scripts (Phase 5) |

Small, hand-written, fully synthetic test fixtures live in `tests/fixtures/` (the only
allowlisted location for CSV files).

## Rules

1. Originals are **never edited**. Any cleaning writes a new file under `data/processed/`.
2. Direct identifiers (names, PRNs, e-mails, phone numbers) are **never** model inputs and are
   not written to logs, reports or exports.
3. Public datasets are used only after their origin, licence and structure are checked
   (`research/DATASET_RESEARCH.md`), and are cited wherever results are shown.
4. Unofficial mirrors are not used without explicit approval.

## Provenance register

| Dataset | Status | Source | Licence | Notes |
|---------|--------|--------|---------|-------|
| Reference CSV (`scholartrack_synthetic_200_students_PRN_sorted.csv`) | **Received** 2026-10-16 from `main` (commit `fa4f9f7`); local read-only copy, SHA-256 `2746d8466cbe93159307b2f178f6894f442226b218162ea5a876d38b5201559a` | You (synthetic, college-style format) | n/a | Schema reference only; identities never copied into generated data. See `reports/data_quality_report.md` |
| UCI 697 — Predict Students' Dropout and Academic Success | Proposed, not downloaded | <https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success> | CC BY 4.0 (to be confirmed at source) | Download host currently blocked by the environment's network policy |
| OULAD | Optional stretch, not downloaded | <https://analyse.kmi.open.ac.uk/open-dataset> | CC BY 4.0 | Download host currently blocked |
| ScholarTrack synthetic dataset | Generated (generator v1, seed 20261016): 1,200 students, 8,561 student-semesters, 88,452 subject results | `scripts/generate_synthetic_data.py` | Ours | Fully synthetic (`SYN…` identifiers); design in `reports/synthetic_data_design.md`, statistics in `reports/synthetic_data_profile.md`; file hashes in `data/synthetic/manifest.json` |

## How to regenerate data

```bash
# Synthetic dataset (Phase 4): rewrites data/synthetic/ and reports/synthetic_data_profile.md
uv run python scripts/generate_synthetic_data.py

# Data-quality report for the reference CSV (Phase 2): needs the file in data/reference/
uv run python scripts/inspect_reference_data.py
```

Both commands are deterministic: re-running them produces byte-identical outputs.

### What `data/synthetic/` contains

| File | Contents | May a model read it? |
|------|----------|----------------------|
| `semesters.csv`, `subjects.csv` | Internal representation (same columns the adapter produces) | Yes (via the feature pipeline) |
| `college_format/<year>_sem<n>.csv` | The same records in the reference CSV layout, one file per academic year and semester | Yes (via loader + adapter); also demo upload files |
| `students.csv` | Synthetic PRN and name for each student key | No — identifiers |
| `latent_truth.csv` | Hidden generating variables (ability, trend, shocks, attrition) | **Never** — teaching and analysis only |
| `manifest.json` | Seed, configuration hashes, row counts and SHA-256 of every file | — |
