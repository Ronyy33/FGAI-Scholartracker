# Technical Walkthrough

A guided tour of the code, in the order data flows through it. Read it next to the source;
every step names the file and function to open. Commands assume `uv` (see README).

## 0. Map of the repository

```text
configs/            YAML: curriculum, target, generator, features, experiment, evaluation,
                    risk bands, recommendations — behaviour you can change without code
src/scholartrack/
  data/             loader → validation → adapter (college CSV → internal tables)
  synthetic/        the synthetic data generator
  targets.py        the label rule
  features/         features, student-grouped split, preprocessor
  models/           baselines, networks, save/load
  training/         data preparation, training loop, experiment runner
  evaluation/       metrics, calibration, thresholds, plots, runner
  risk.py           probability → score → band
  explain.py        concept-group explanations + supportive suggestions
  api/              FastAPI app, scoring service, demo data, settings, history
scripts/            entry points (generate, train, evaluate, final_evaluation, export bundle …)
models/serving/     the frozen, evaluated models the API loads (committed, 44 KB)
reports/            generated, aggregate-only reports and figures
frontend/           React dashboard
tests/              pytest suite
```

## 1. Reading a college CSV safely

`data/loader.py::read_college_csv(source, max_rows, max_bytes)`

- Enforces size and row limits before parsing (uploads can be hostile).
- Strips a byte-order mark; reads everything as text first so that `"007"` or `"12.4abc"` are
  seen as typed, not silently converted.
- Rejects duplicate column names itself, because pandas would rename them to `x.1` quietly.

`data/validation.py::validate_college_frame(frame, curriculum)` returns a `ValidationReport`
of `Issue`s with three severities:

| Severity | Meaning | What happens |
|----------|---------|--------------|
| `error` | row unusable (no PRN, no semester, duplicate student-semester) | row dropped and listed |
| `invalid` | impossible value (attendance 145 %, CIE −5) | value becomes missing — never "repaired" |
| `warning` | suspicious but possible (unknown elective) | value kept, reported |

Ranges come from `configs/curriculum.yaml` (e.g. CIE max 40 for theory subjects), not from
constants in code.

## 2. From wide CSV to internal tables

`data/adapter.py::adapt_college_frame(frame, curriculum)` → `AdaptedData`

- `semesters`: one row per student-semester (SGPA recomputed from subject marks with the
  recovered credits; reported summary columns are *not trusted*).
- `subjects`: one row per subject result (CIE, ETE, termwork, total, attendance, passed).
- PRNs are replaced by opaque keys `S000001…`; the mapping (`id_map`) stays in memory so the
  API can put PRNs back into the response the user asked for.

**Why recompute SGPA?** The reference file's reported values were inconsistent with its marks
(see `reports/data_quality_report.md`); deriving everything from subject marks with one rule
keeps training and serving identical.

## 3. The label

`targets.py` joins each student-semester *t* with semester *t + 1* of the same student and
applies `configs/target.yaml`: positive if any subject in *t + 1* is not passed, or SGPA < 6.0.
Labels are three-valued (1 / 0 / NaN) with a `label_status` explaining NaN (no next semester
yet, student left, incomplete marks). NaN rows are scored but never trained on.

## 4. Synthetic data

`synthetic/generator.py::generate(cfg, curriculum, target)` (the seed is in `configs/synthetic.yaml`)

1. Draw each student's latent ability and engagement (correlated), a trajectory
   (improve / stable / decline) and a cohort.
2. For each semester: latent performance = ability + trend + AR(1) noise + occasional shock;
   attendance depends on engagement; subject marks are drawn around the latent performance and
   clipped to each component's maximum.
3. Some students leave after poor semesters (attrition); some cells are blanked at random
   (MCAR) and some not at random (disengaged students skip `study_hours_per_week`).
4. `to_college_format` writes the same records in the reference CSV layout, one file per
   academic year and semester.

The generator never writes a label. `scripts/generate_synthetic_data.py` writes a manifest
with file hashes; the serving bundle records the manifest hash it was trained on.

## 5. Features

`features/build.py`

- `semester_features(semesters, subjects)` → 15 numbers per student-semester, all computable
  from one college-format row plus `prior_sgpa` (so a single uploaded semester is enough to
  score).
- `build_sequences(...)` → for each student-semester *t*, the feature vectors of semesters
  1…*t* (for the LSTM).
- Tested for **no look-ahead**: deleting every semester after *t* leaves the features and
  sequence for *t* unchanged (`tests/test_features.py`).

`features/split.py` assigns whole students to train / validation / test (70/15/15) within each
admission cohort. `features/preprocess.py::Preprocessor` adds `_missing` flags, imputes training
medians and standardises with training means and standard deviations; `to_json` / `from_json`
store it readably.

## 6. Models and training

- `models/baselines.py`: `MajorityBaseline`; `LogisticModel` (scikit-learn for fitting, saved as
  plain JSON coefficients).
- `models/networks.py`:
  - `MLP`: `Linear → ReLU → Dropout` blocks then one output logit.
  - `LSTMClassifier`: sequences are padded to a batch, then `pack_padded_sequence` so each
    student's LSTM stops at their own last semester; the final hidden state → dropout →
    linear → logit.
- `training/loop.py`: AdamW, binary cross-entropy on logits, early stopping on validation log
  loss, gradient clipping for the LSTM, fixed seeds.
- `training/experiment.py`: small grid per family (`configs/experiment.yaml`), choose by
  validation log loss, retrain the chosen networks with 3 seeds to measure seed variation,
  save everything under `artifacts/runs/<run_id>/` with a manifest.

The test partition is loaded only by `scripts/final_evaluation.py --stage test --final`, which
refuses to run twice for the same run.

## 7. Evaluation

`evaluation/`

- `metrics.py`: ROC-AUC, PR-AUC, log loss, Brier, ECE, reliability bins.
- `calibration.py`: compares no calibration with Platt scaling by student-grouped
  cross-validation on validation data.
- `classification.py::choose_threshold`: the F1-maximising threshold over the exact predicted
  values (a bug with rounded candidates was found and fixed with a regression test).
- `runner.py`: predictions per partition, Top-K capture, bootstrap confidence intervals by
  student (resampling students, not rows, because rows of one student are correlated).

## 8. Risk score and explanation

- `risk.py`: `risk_score = floor(100 p + 0.5)`; `risk_band` reads `configs/risk_bands.yaml`.
- `explain.py::explain(...)`: for each concept group, replace the student's values in that
  group with a typical student's (training medians), re-score, and report the change in
  points. Groups are used instead of single indicators because correlated indicators (SGPA,
  marks, failures) share credit and individually can show misleading signs.
- `explain.py::recommend(raw, rules)`: rule-based, supportive suggestions keyed on *observed*
  values (e.g. attendance below the 75 % guideline), with a fixed disclaimer.

## 9. Serving

`scripts/export_serving_bundle.py` copies the evaluated models into `models/serving/` with a
`bundle.json` that lists SHA-256 hashes, thresholds, test metrics, versions and the dataset
manifest hash.

`api/service.py::ScoringService` verifies every hash at start-up, then `score(semesters,
subjects)` scores each student's latest semester with all three models (the default model's
score becomes the headline), adds explanations and suggestions.

`api/app.py::create_app(settings)` wires the endpoints (see `docs/ARCHITECTURE.md` §4):
uploads are limited (2 MB, 5,000 rows, `.csv` only), errors share one JSON shape, logs record
counts only, and the built dashboard is served from `frontend/dist` with a fallback to
`index.html` for client-side routes.

`api/demo.py::DemoData` loads the synthetic dataset (only if its manifest hash matches the
bundle), picks "currently enrolled" students (latest record in the newest academic year,
semester < 8) and computes aggregates with groups smaller than 10 suppressed.

## 10. Frontend

`frontend/src/`

- `lib/api.ts`: `apiFetch` (turns the API's error shape into `ApiError`), `useApi` (loading /
  error / data derived from the request key; no state set inside effects).
- `lib/palette.ts`: chart colours read from CSS variables, so light and dark mode switch
  together; models are distinguished by colour **and** dash pattern.
- `components/ui.tsx`: cards, stat tiles, `BandChip` (icon + text + colour, never colour
  alone), loading / error / empty states, the synthetic-data banner.
- `pages/`: one file per page. `BatchPage.previewCsv` shows the first rows of an upload without
  any column whose name contains "name".

## 11. Try it yourself

```bash
uv run python scripts/generate_synthetic_data.py
uv run pytest -q
uv run uvicorn scholartrack.api.app:app --port 8000      # then open /api/docs
curl -s -X POST localhost:8000/api/predict -H 'Content-Type: application/json' \
     -d @examples/predict_request.json | python -m json.tool | head -40
```

Exercises that build understanding:
1. Change `configs/risk_bands.yaml` to Low 0–19 / Moderate 20–49 / High 50–100, restart the API
   and watch the Overview change — no retraining needed.
2. In `examples/predict_request.json`, lower one subject's `ETE_marks` below 40 % of its
   maximum and re-post: the "Current-semester results" effect grows and the failed-subject
   suggestion appears.
3. Set `prior_sgpa` to 12.4: the response lists an `invalid` issue and the value is treated as
   missing.
