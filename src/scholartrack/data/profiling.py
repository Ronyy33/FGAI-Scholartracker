"""Helpers for profiling a dataset (used by the data-quality report)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CreditInference:
    """Result of recovering subject credits from a credit-weighted summary column."""

    weights: pd.Series  # least-squares weights (sum ~ 1)
    credits: pd.Series  # weights rescaled so the smallest is 1, rounded to whole numbers
    n_rows: int
    max_abs_residual: float  # using the rounded credits
    rmse: float


def infer_credits(subject_values: pd.DataFrame, summary: pd.Series) -> CreditInference:
    """Recover relative credits from ``summary = sum(credit_i * value_i) / sum(credit_i)``.

    Solves the least-squares problem summary ~ values @ w on complete rows. If the summary
    really is a credit-weighted mean, ``w_i = credit_i / total_credits``, so dividing by the
    smallest weight gives credits relative to the lightest subject. The rounded credits are
    then checked by recomputing the summary and measuring the residuals.
    """
    complete = subject_values.notna().all(axis=1) & summary.notna()
    x = subject_values[complete].to_numpy(dtype=float)
    y = summary[complete].to_numpy(dtype=float)
    if len(y) < x.shape[1]:
        raise ValueError("Not enough complete rows to infer credits.")
    w, *_ = np.linalg.lstsq(x, y, rcond=None)
    weights = pd.Series(w, index=subject_values.columns)
    credits = (weights / weights.min()).round().astype(int)
    fitted = x @ (credits.to_numpy() / credits.sum())
    residuals = y - fitted
    return CreditInference(
        weights=weights,
        credits=credits,
        n_rows=int(len(y)),
        max_abs_residual=float(np.abs(residuals).max()),
        rmse=float(np.sqrt(np.mean(residuals**2))),
    )
