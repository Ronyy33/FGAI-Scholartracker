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
