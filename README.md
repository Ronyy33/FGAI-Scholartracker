# ScholarTrack — Detecting Students at Risk of Falling Behind

**Project Sheet:** PS 6 · **Techniques:** Multilayer Perceptron (MLP) and Long Short-Term Memory (LSTM)

[![CI](https://github.com/Ronyy33/FGAI-Scholartracker/actions/workflows/ci.yml/badge.svg?branch=claude/modest-bell-box1op)](https://github.com/Ronyy33/FGAI-Scholartracker/actions/workflows/ci.yml)

> **Status: complete and deployment-ready; not yet deployed** (deploying needs your hosting
> account — see [`docs/DEPLOYMENT_GUIDE.md`](docs/DEPLOYMENT_GUIDE.md)). All results come from a
> **synthetic** dataset and describe agreement with a documented synthetic process, not
> real-world accuracy. Current state: [`PROJECT_STATUS.md`](PROJECT_STATUS.md).

## What it does

At the end of each semester ScholarTrack estimates the probability that a student's **next**
semester will be an academic-difficulty semester (at least one subject not passed, or SGPA
below 6.0 — illustrative, configurable thresholds). It shows:

- an **Academic Risk Score** (0–100 = calibrated probability × 100) and a band
  (Low 0–29 · Moderate 30–59 · High 60–100, demonstration defaults);
- the **main contributing factors**, in risk points versus a typical student;
- **supportive suggestions** (mentoring, tutoring, study planning) — never penalties.

ScholarTrack is **decision support**. It is not a basis for punishment, exclusion or academic
penalties, and it does not diagnose motivation, ability or personal circumstances.

## Results at a glance (test set, scored once)

| Model | ROC-AUC [95 % CI] | PR-AUC | Brier |
|-------|-------------------|-------:|------:|
| Majority baseline | 0.500 | 0.204 | 0.162 |
| **Logistic regression (default)** | 0.878 [0.840, 0.910] | 0.705 | 0.101 |
| MLP | 0.881 [0.844, 0.913] | 0.707 | 0.101 |
| LSTM | 0.880 [0.842, 0.912] | 0.714 | 0.100 |

The three learned models are statistically tied, so the pre-registered rule chose the simplest.
The LSTM cannot beat the snapshot models here because, in this synthetic data, history adds no
information beyond the latest semester. Details: [`reports/evaluation_report.md`](reports/evaluation_report.md).

## Quick start

### Docker (everything in one container)

```bash
docker build -t scholartrack .
docker run --rm -p 8000:8000 scholartrack
# dashboard: http://localhost:8000   ·   API docs: http://localhost:8000/api/docs
```

### Development

Requires Python ≥ 3.11 with [uv](https://docs.astral.sh/uv/), and Node 22.

```bash
uv sync                                              # Python environment from uv.lock
uv run python scripts/generate_synthetic_data.py     # demo dataset (deterministic, ~5 s)
cd frontend && npm ci && npm run build && cd ..      # dashboard → frontend/dist
uv run uvicorn scholartrack.api.app:app --port 8000  # API + dashboard on :8000
```

Frontend with hot reload: keep the API running and use `cd frontend && npm run dev`.

### Checks

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest -q                       # 162 tests; 4 skip without the private reference CSV
cd frontend && npm run typecheck && npm run lint && npm test
```

### Reproduce the models (optional)

```bash
uv run python scripts/train.py                              # all models, ~1 min on CPU
uv run python scripts/final_evaluation.py --stage validation
# the test stage (--stage test --final) runs once per training run, by design
uv run python scripts/export_serving_bundle.py
```

## Using the API

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/api/predict -H 'Content-Type: application/json' \
     -d @examples/predict_request.json
curl -s -X POST localhost:8000/api/predict/batch -F "file=@your_college_format.csv"
```

Uploads: `.csv` in the college format, ≤ 2 MB, ≤ 5,000 rows, processed in memory and not
stored. Every endpoint is listed in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#4-api-surface)
and documented interactively at `/api/docs`.

## The dashboard

| Page | Shows |
|------|-------|
| Overview | Records processed, students scored, flagged count, band distribution, model status |
| Student risk | Sortable, filterable list → student page with score, band, three model scores, contributing factors, suggestions, semester history |
| Batch prediction | CSV upload, preview (names hidden), validation report, sortable predictions, CSV download |
| Analytics | Risk distribution, attendance vs marks, SGPA by cohort, subject results (groups < 10 suppressed) |
| Model evaluation | Metrics with confidence intervals, ROC / PR / calibration curves, confusion matrix, Top-K capture, band check |
| About | Method, data, limitations, responsible use |

## Architecture

```text
configs/*.yaml ──► synthetic generator ──► data/synthetic (Git-ignored)
                                               │
                     features ─► split by student ─► train LR · MLP · LSTM
                                               │
             validation: calibration, thresholds, model choice (committed first)
                                               │
                     test set scored once ─► reports/  ─► models/serving/ (frozen, hashed)
                                                               │
                         FastAPI (src/scholartrack/api) ◄──────┘ ──► React dashboard
```

Full diagrams: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Documentation

| Document | Purpose |
|----------|---------|
| [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md) | Full report, results, limitations, demo checklist |
| [`docs/TECHNICAL_WALKTHROUGH.md`](docs/TECHNICAL_WALKTHROUGH.md) | Guided tour of the code, with exercises |
| [`docs/VIVA_PREPARATION.md`](docs/VIVA_PREPARATION.md) | Likely examiner questions and answers |
| [`docs/LEARNING_NOTES.md`](docs/LEARNING_NOTES.md) | Concepts per phase, common mistakes |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Components, request flow, API surface |
| [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) | Every feature and excluded column (generated from code) |
| [`docs/TARGET_DEFINITION.md`](docs/TARGET_DEFINITION.md) | What "falling behind" means and why |
| [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) | Experiments, including what did not work |
| [`docs/DEPLOYMENT_GUIDE.md`](docs/DEPLOYMENT_GUIDE.md) | Docker, Hugging Face Spaces, Render, verification |
| [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) | Original plan and milestones |
| [`research/`](research/) | Public dataset research and comparison |
| [`reports/`](reports/) | Generated reports and figures (aggregate only) |
| [`TASKS.md`](TASKS.md) · [`PROJECT_STATUS.md`](PROJECT_STATUS.md) · [`IMPLEMENTATION_LOG.md`](IMPLEMENTATION_LOG.md) | Checklist, current state, chronological log |

## Repository structure

```text
.
├── configs/              # curriculum, target, generator, features, experiment, evaluation, bands, suggestions
├── data/                 # Git-ignored except README (repository is public)
├── docs/                 # report, walkthrough, viva, architecture, deployment, learning notes …
├── examples/             # fictional API request
├── frontend/             # React + TypeScript + Tailwind + Recharts dashboard
├── models/serving/       # frozen, evaluated model bundle (44 KB) loaded by the API
├── reports/              # generated reports and figures
├── research/             # dataset research
├── scripts/              # generate, train, evaluate, final evaluation, export bundle, diagnostics
├── src/scholartrack/     # data, synthetic, features, models, training, evaluation, explain, api
├── tests/                # pytest suite
├── Dockerfile · render.yaml · .env.example
└── .github/workflows/ci.yml
```

## Data and privacy

- The GitHub repository is **public**. No student-level data — real or synthetic — is
  committed; tests fail if a data file is added.
- Names, PRNs and other direct identifiers are never model inputs; the upload preview hides
  name columns; logs contain counts only; analytics suppress groups smaller than 10.
- Your reference CSV stays local in `data/reference/` and is never modified in place.
- The app has **no authentication**, so a public demo must only serve synthetic data.

## Responsible use

Risk scores are estimates with uncertainty. They should prompt supportive conversations,
tutoring or advising — never automatic decisions. A real deployment would need local data with
observed outcomes, re-validation, a fairness review, institutional authentication and approval.
