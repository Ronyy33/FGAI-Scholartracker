# ScholarTrack — Detecting Students at Risk of Falling Behind

**Project Sheet:** PS 6 · **Techniques:** Multilayer Perceptron (MLP) and Long Short-Term Memory (LSTM)

> **Project status: Milestone 1 (planning and dataset research).**
> No model has been trained yet and the web application does not exist yet. This README
> describes only what is implemented; planned work is labelled as planned. See
> [`PROJECT_STATUS.md`](PROJECT_STATUS.md) for the current state.

## Problem statement

Given the academic information available about a student at a point in time, estimate the
probability that the student will experience a defined academic difficulty in a future
assessment period, and present it as an **Academic Risk Score (0–100)** with understandable
contributing factors and supportive suggestions.

ScholarTrack is **decision support**. It is not a basis for punishment, exclusion or academic
penalties, and it does not diagnose motivation, ability or personal circumstances.

## What exists today

| Item | Location |
|------|----------|
| Project plan, architecture, stack, target and risk-score design | [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) |
| Public dataset research with sources and verification levels | [`research/DATASET_RESEARCH.md`](research/DATASET_RESEARCH.md) |
| Dataset comparison and recommendation | [`research/DATASET_COMPARISON.md`](research/DATASET_COMPARISON.md) |
| Data-handling rules and provenance register | [`data/README.md`](data/README.md) |
| Learning notes (per phase) | [`docs/LEARNING_NOTES.md`](docs/LEARNING_NOTES.md) |
| Repository-hygiene tests (no data or secrets committed) | [`tests/test_repo_hygiene.py`](tests/test_repo_hygiene.py) |
| Task checklist · status · chronological log | [`TASKS.md`](TASKS.md) · [`PROJECT_STATUS.md`](PROJECT_STATUS.md) · [`IMPLEMENTATION_LOG.md`](IMPLEMENTATION_LOG.md) |

## Planned components

Planned, **not yet implemented**: data validation and schema adapters, synthetic data
generator, feature pipeline, baseline / MLP / LSTM models, calibration and evaluation suite,
FastAPI backend, React + TypeScript dashboard, CI, Docker image, deployment. The milestone
plan is in [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md#2-milestones).

## Repository structure (current)

```text
.
├── data/                 # ignored by Git except README + placeholders (repository is public)
├── docs/                 # plan and learning notes
├── research/             # dataset research and comparison
├── src/scholartrack/     # Python package (empty skeleton for now)
├── tests/                # pytest suite
├── pyproject.toml        # Python project + tool configuration (uv)
├── IMPLEMENTATION_LOG.md
├── PROJECT_STATUS.md
└── TASKS.md
```

## Development setup

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # create .venv and install the locked dev tools
uv run pytest           # run the test suite
uv run ruff check .     # lint
uv run ruff format --check .
```

## Data and privacy

- The GitHub repository is **public**. No student-level data — real or synthetic — is committed.
- Names, PRNs and other direct identifiers are never used as model inputs.
- Your reference CSV stays local in `data/reference/` and is never modified in place.

## Responsible use

Risk scores are estimates with uncertainty. They should prompt supportive conversations,
tutoring or advising — never automatic decisions. Any real deployment at a college would need
local data, validation, a privacy review and institutional approval first.
