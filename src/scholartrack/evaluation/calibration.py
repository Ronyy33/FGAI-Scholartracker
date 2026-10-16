"""Calibration: do predicted probabilities match observed frequencies?

If, among students given ≈ 30 % risk, about 30 % actually have a difficulty semester, the model
is *calibrated* and its scores can be read as probabilities. We measure this with a reliability
table and the expected calibration error (ECE), and optionally correct it with Platt scaling —
a one-feature logistic regression on the model's logit — fitted on validation data only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Equal-width probability bins: count, mean predicted probability, observed rate."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        rows.append(
            {
                "bin": f"{edges[b]:.1f}–{edges[b + 1]:.1f}",
                "count": int(mask.sum()),
                "mean_predicted": float(p[mask].mean()) if mask.any() else np.nan,
                "observed_rate": float(y[mask].mean()) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    table = reliability_table(y, p, n_bins).dropna()
    weights = table["count"] / table["count"].sum()
    return float((weights * (table["mean_predicted"] - table["observed_rate"]).abs()).sum())


@dataclass(frozen=True)
class PlattCalibrator:
    slope: float
    intercept: float

    @classmethod
    def fit(cls, p: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        model = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        return cls(float(model.coef_[0, 0]), float(model.intercept_[0]))

    def transform(self, p: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.slope * _logit(p) + self.intercept)))


def choose_calibration(
    y: np.ndarray, p: np.ndarray, groups: np.ndarray, folds: int, min_improvement: float
) -> dict:
    """Compare 'none' vs Platt by cross-validation *by student* inside the validation set."""
    raw, platt = np.zeros(len(y)), np.zeros(len(y))
    for fit_idx, eval_idx in GroupKFold(n_splits=folds).split(p, y, groups):
        raw[eval_idx] = p[eval_idx]
        platt[eval_idx] = PlattCalibrator.fit(p[fit_idx], y[fit_idx]).transform(p[eval_idx])
    loss_raw = float(log_loss(y, np.clip(raw, EPS, 1 - EPS)))
    loss_platt = float(log_loss(y, np.clip(platt, EPS, 1 - EPS)))
    use_platt = loss_raw - loss_platt > min_improvement
    result = {
        "method": "platt" if use_platt else "none",
        "cv_log_loss_none": loss_raw,
        "cv_log_loss_platt": loss_platt,
    }
    if use_platt:
        calibrator = PlattCalibrator.fit(p, y)
        result.update(slope=calibrator.slope, intercept=calibrator.intercept)
    return result


def apply_calibration(p: np.ndarray, choice: dict) -> np.ndarray:
    if choice["method"] == "none":
        return np.asarray(p, dtype=float)
    return PlattCalibrator(choice["slope"], choice["intercept"]).transform(p)
