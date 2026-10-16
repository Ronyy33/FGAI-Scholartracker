# Experiment Log

Chronological record of actual experiments: configuration, data version, seeds, results,
problems and decisions. Numbers are copied from generated files named next to them; those
files are the source of truth.

**Conventions.** All experiments so far use **synthetic data** (Track S); results measure how
well models recover the documented synthetic label process, not real-world predictive ability.
Model settings are chosen on the **validation** partition. The **test** partition is reserved
for one final evaluation in Phase 7 and is guarded in code (`scripts/evaluate.py --final`).

---

## EXP-001 — Baselines, MLP and LSTM (Phase 6) · 2026-10-16

| Item | Value |
|------|-------|
| Official run | `20261016T200932Z-8acaad7` (code commit `8acaad7`, clean working tree) |
| Dataset | synthetic generator v1, seed 20261016; `data/synthetic/manifest.json` SHA-256 `3bf92f541da9…` |
| Split | by student within admission cohort, seed 20261017: 5,112 / 1,088 / 1,100 labelled rows (train / validation / test); positive rates 20.7 % / 19.3 % / 20.4 % |
| Inputs | 15 snapshot features + 3 missing indicators (`docs/DATA_DICTIONARY.md`); LSTM: the same vector per semester, sequences of 1–8 semesters |
| Configuration | `configs/experiment.yaml`, experiment seed 20261018; selection = lowest validation log loss |
| Software | Python 3.13.16, PyTorch 2.14.1 (PyPI build, run on CPU), scikit-learn 1.9.1, NumPy 2.5.4, pandas 3.0.6 |
| Test partition used? | **No** |

### Results (validation; from `reports/training_report.md`)

| Model | Chosen setting | Log loss ↓ | Brier ↓ | ROC-AUC ↑ | PR-AUC ↑ |
|-------|----------------|-----------:|--------:|----------:|---------:|
| Majority baseline | — | 0.4912 | 0.1560 | 0.5000 | 0.1930 |
| Logistic regression | C = 0.1, no class weights | **0.2615** | **0.0798** | **0.9256** | **0.7869** |
| MLP | one hidden layer of 32, dropout 0.1 (best epoch 42) | 0.2636 | 0.0800 | 0.9228 | 0.7837 |
| LSTM | hidden size 32, dropout 0.3 (best epoch 17) | 0.2641 | 0.0798 | 0.9227 | 0.7774 |

Seed variation of the chosen networks (3 seeds, mean ± sd): MLP log loss 0.2633 ± 0.0007,
ROC-AUC 0.9228 ± 0.0001; LSTM log loss 0.2646 ± 0.0005, ROC-AUC 0.9222 ± 0.0005.

### Observations

1. **All learned models far exceed the majority baseline, and are nearly tied with each
   other.** Differences between logistic regression, MLP and LSTM are ≤ 0.003 in log loss and
   ≤ 0.01 in PR-AUC — larger than seed-to-seed variation, but small. Whether they are
   meaningful needs confidence intervals (planned for Phase 7, bootstrap on the test set).
2. **Class weighting hurts probabilities, not ranking.** `class_weight="balanced"` gave almost
   the same ROC-AUC (≈ 0.925) but log loss ≈ 0.34 instead of ≈ 0.26, because it inflates
   predicted risk. Unweighted models were therefore chosen; imbalance will be handled by
   choosing the decision threshold on validation data (Phase 7).
3. **Early stopping matters for the LSTM.** Its validation loss was lowest at epoch 17 and then
   rose while training loss kept falling (overfitting); the best-epoch weights are restored.
4. **Why the LSTM does not beat the snapshot models — diagnostic** (`reports/signal_diagnostic.md`,
   `scripts/diagnose_signal.py`; validation only):

   | Logistic regression inputs | Log loss | ROC-AUC |
   |----------------------------|---------:|--------:|
   | 15 snapshot features | 0.2619 | 0.9254 |
   | snapshot + history summaries | 0.2619 | 0.9244 |
   | ORACLE: generator's hidden state (unavailable in reality) | 0.2421 | 0.9361 |

   History summaries add nothing, so in this synthetic data the past beyond the current and
   previous semester carries little extra information — no architecture can exploit what is
   not there. Even a model that knew the hidden state would reach only ROC-AUC ≈ 0.936; the
   rest is next-semester randomness. The observable models are close to that ceiling.
   **The generator was not changed to make the LSTM look better**; that would be tuning the
   data to the preferred answer.

### Reproducibility checks

- Re-running `scripts/train.py` gave identical metrics for all 16 grid settings and all extra
  seeds (runs `20261016T200455Z-21230cb`, `20261016T200747Z-21230cb` — both from an uncommitted
  working tree during development — and the official run).
- `scripts/evaluate.py` reloads the saved models from disk and reproduces the training-time
  validation predictions to within 1.1 × 10⁻¹⁶.
- `scripts/evaluate.py --partition test` without `--final` refuses to run.

### Problems and fixes

- PyTorch's CPU-only wheel host is blocked here; the PyPI build (bundling unused GPU libraries)
  was installed instead. It runs on CPU; deployment size is handled in Phase 14.
- A PyTorch warning about converting a gradient-tracking tensor (`float(loss)`) was fixed with
  `loss.item()`.
- The training report printed `2.5000` seconds because the table helper re-parsed
  already-formatted text; fixed in `reporting.fmt`, with tests.

### Decision status

**Provisional.** Logistic regression has the best validation scores and is the most
interpretable; the MLP and LSTM (required by PS 6) are within a few thousandths. The final model
will be chosen in Phase 7 using calibration, threshold behaviour, interpretability and the
documented trade-offs — then evaluated once on the test set.

---

## EXP-002 — Calibration, thresholds, model choice and the one-time test evaluation (Phase 7) · 2026-10-16

| Item | Value |
|------|-------|
| Run | `20261016T200932Z-8acaad7` (same models as EXP-001; nothing retrained) |
| Settings | `configs/evaluation.yaml`, `configs/risk_bands.yaml` |
| Pre-registration | validation-stage choices committed and pushed in `776700c` **before** the test set was scored (`reports/model_selection.json`) |
| Test evaluations | exactly one (recorded in the run manifest; a second attempt is refused) |

### Decisions on validation (frozen before testing)

| Model | Calibration (CV by student inside validation) | Threshold (max F1) | Val log loss | Val ECE |
|-------|---------------------------------------------|-------------------:|-------------:|--------:|
| Logistic regression | none (Platt: 0.2623 vs none 0.2615) | 0.306 | 0.2615 | 0.0193 |
| MLP | none (0.2647 vs 0.2636) | 0.351 | 0.2636 | 0.0136 |
| LSTM | none (0.2645 vs 0.2641) | 0.349 | 0.2641 | 0.0164 |

Selection rule: lowest validation log loss; within 0.005 counts as tied; ties go to the simpler
model. All three were tied → **logistic regression chosen**.

### Test results (from `reports/evaluation_report.md`; 1,100 rows, 179 students, positive rate 20.4 %)

| Model | ROC-AUC [95 % CI] | PR-AUC [95 % CI] | Brier | F1 at frozen threshold | Top-10 % capture |
|-------|-------------------|------------------|------:|-----------------------:|-----------------:|
| Logistic regression (chosen) | 0.878 [0.840, 0.910] | 0.705 [0.587, 0.793] | 0.1013 | 0.621 | 39.7 % (89 of 224) |
| MLP | 0.881 [0.844, 0.913] | 0.707 [0.596, 0.792] | 0.1012 | 0.616 | 40.6 % |
| LSTM | 0.880 [0.842, 0.912] | 0.714 [0.602, 0.798] | 0.1001 | 0.630 | 41.1 % |
| Majority | 0.500 | 0.204 | 0.1622 | 0.338 | 11.2 % |

- Paired differences (chosen − other, bootstrap by student) all have 95 % intervals containing
  0: **the test set cannot distinguish the three learned models.**
- Calibration holds on test: risk bands Low / Moderate / High contain 812 / 157 / 131 rows with
  observed difficulty rates 8.3 % / 36.3 % / 76.3 % — each inside its band's score range.
- Chosen model at threshold 0.306: TP 155, FP 120, FN 69, TN 756 (recall 0.69, precision 0.56;
  25 % of students flagged).

### The validation–test gap

Test scores are lower than validation for every model (LR ROC-AUC 0.926 → 0.878). Analysis
(`scripts/diagnose_split_variation.py` → `reports/split_variation.md`): the 95 % intervals
overlap (validation [0.898, 0.947]); 7-fold cross-validation by student over all 1,183 students
gives fold ROC-AUC 0.896–0.934 (mean 0.915) — partitions of ~180 students vary this much by
chance. Errors are not concentrated in a few test students (worst 10 = 20 % of squared error vs
27 % in validation); test students had slightly more synthetic shocks (0.385 vs 0.352 per
student). Conclusion: **sampling variation**, not overfitting (all models dropped equally,
including the barely tuned logistic regression). Expected ROC-AUC on new synthetic students:
≈ 0.88–0.93. The published test numbers are reported as they came out.

### Problems and fixes

- Threshold search crashed on the constant majority baseline: candidates rounded to 6 decimals
  could exceed every prediction and flag nobody. Fixed (exact values) with a regression test —
  found during the validation stage, before any test scoring.
- Two figure-layout defects found by looking at the rendered PNGs (clipped confusion-matrix label,
  gridlines over bars). Fixed; figures redrawn by `--stage redraw`, which recomputes test metrics
  with the frozen models and refuses unless they equal the published values exactly (they did).

### Final model decision

**Logistic regression** is the default model behind the Academic Risk Score: tied on every
metric with the MLP and LSTM, best validation log loss, well calibrated without correction, and
its coefficients can be shown and explained to students and faculty. The MLP and LSTM remain
trained, evaluated and selectable (`configs/` will expose the default model for the API), so the
application can display all three side by side.

---

## EXP-003 — Per-student explanations (Phase 8)

**Question.** How should the app explain one student's score so that the explanation is both
faithful to the model and not misleading?

**Attempt 1 — per-indicator perturbation (rejected).** Replacing one indicator at a time with
its training median and measuring the change in risk produced counter-intuitive signs for
correlated indicators: e.g. "subjects not passed" could appear to *lower* risk because current
SGPA and credit-weighted marks already carry the same information (logistic coefficient for
`n_failed` is −0.142 while the three together clearly raise risk). Showing that to a mentor
would be misleading, so this design was dropped before any UI was built on it.

**Attempt 2 — concept groups (adopted).** Five groups (current-semester results, previous
semester and trend, attendance, self-reported study time, stage of the programme) are replaced
*together* with a typical student's values; the change in risk points is reported, with the
group's indicators and their typical values. Implemented in `src/scholartrack/explain.py`;
tested in `tests/test_explain.py`.

**Global check** (`scripts/explain_global.py` → `reports/explanations.md`, validation data):
- For the 109 validation students with a failed subject, the results group raises the
  estimate in 94 % (LR) / 96 % (MLP) of cases, mean +56 points. As expected.
- For the 283 students with attendance < 75 %, the attendance group *lowers* the estimate on
  average (−4.3 points LR, −4.6 MLP). Cause: in the generator, engagement drives both
  attendance and marks, so at equal marks lower attendance indicates higher latent ability
  (*explaining away*). The models are faithful to the data; the association is not causal.

**Decision.** Show the attendance effect honestly with a note; attendance *suggestions* come
from the 75 % guideline in `configs/recommendations.yaml`, never from the model's sign.
Permutation importance (drop in validation ROC-AUC) is reported per indicator for the global
view, where shared credit is expected and explained.
