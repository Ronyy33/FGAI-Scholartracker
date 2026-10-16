"""Threshold-free metrics used for model selection (the full suite comes in Phase 7).

- **Log loss** (binary cross-entropy): penalises confident wrong probabilities; a *proper*
  scoring rule, so it rewards honest probabilities. Lower is better. Used for selection.
- **Brier score**: mean squared error of the probability. Lower is better.
- **ROC-AUC**: probability that a random positive gets a higher score than a random negative.
  0.5 = random ranking. Insensitive to class imbalance, which can make it look optimistic.
- **PR-AUC** (average precision): summarises precision vs recall; its "random" level equals the
  positive rate, so it is more informative when positives are rare.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def probability_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    both_classes = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if both_classes else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if both_classes else float("nan"),
    }
