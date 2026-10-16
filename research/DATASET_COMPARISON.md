# Dataset Comparison

Companion to `DATASET_RESEARCH.md`, which holds the sources and verification levels behind
each cell. Ratings are judgements for **this project's** goals, not general quality scores.

## Side-by-side

| Criterion | UCI 697 Dropout & Success | OULAD | UCI 320 Student Perf. | UCI 467 Academics Perf. | UCI 856 HE Perf. Eval. | PredictGrad | Kaggle risk sets | Our synthetic (planned) |
|-----------|---------------------------|-------|-----------------------|-------------------------|------------------------|-------------|------------------|-------------------------|
| Level | University (polytechnic) | University (distance) | Secondary school | College (India, reportedly) | University | Engineering college (India) | Unknown | Engineering college format |
| Rows | 4,424 students | 32,593 students; 10.7 M activity rows | 649 + 395 | 300 | 145 | 905 | Varies | 5,000–10,000 student-semesters (target) |
| Observed outcome | **Yes** (Dropout / Enrolled / Graduate) | **Yes** (final result) | Yes (final grade) | Concurrent grade only | Concurrent grade only | No (model-derived flag) | Undocumented | **No** — documented synthetic rule |
| Time steps | 2 semesters | Weekly / daily | 3 grading periods | 1 | 1 | 3 semesters | Varies | 4–8 semesters per student |
| Matches college schema (SGPA, CIE/ETE/TW, attendance) | Partial (semester units and grades) | Poor | Poor | Partial (internal + end-semester %) | Poor | Good | Unknown | **Designed to match** |
| Licence | CC BY 4.0 (V3) | CC BY 4.0 (V1/V2) | CC BY 4.0 (V2) | CC BY 4.0 (V2) | CC BY 4.0 (V2) | Repo MIT, data unstated | Unverified | Ours |
| Sensitive attributes | Gender, nationality, marital, parents' jobs | Gender, age band, IMD band, disability | Alcohol, family, relationships | Reportedly caste, family income | Survey items | Inferred gender and religion | Unknown | None (synthetic IDs only) |
| Reproducible download | Yes (UCI; blocked here) | Yes (OU site; blocked here) | Yes (UCI; blocked here) | Yes (UCI; blocked here) | Yes (UCI; blocked here) | Yes (GitHub) | Kaggle API with account | Yes (seeded script) |
| MLP suitability | **High** | High (aggregated) | Low–medium | Low | Very low | Medium | — | High |
| LSTM suitability | Low (2 steps) | **High** | Not justified | None | None | Low–medium | — | High (by design) |
| Verdict | **Recommended: real-outcome benchmark** | **Optional stretch: real LSTM** | Teaching example only | Reference for CIE/ETE realism only | Rejected | **Rejected** (provenance, ethics, no outcome) | **Rejected** (provenance) | **Primary for the app** |

## Why not just use one dataset?

- Using **only synthetic data** makes the project easy to align with the college format, but
  every metric would measure how well the model recovers our own label rule.
- Using **only UCI 697** gives real outcomes, but the web app could not accept your college's
  CSV format, and the LSTM would rest on two time steps.
- The **two-track** plan keeps each claim honest: real-outcome evidence comes from UCI 697;
  the college-format workflow and the sequence model are demonstrated on clearly labelled
  synthetic data.

## Decision status

**Pending your approval** — see `docs/PROJECT_PLAN.md` §5 and `PROJECT_STATUS.md`.
