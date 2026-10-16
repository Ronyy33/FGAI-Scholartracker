"""Credit inference must recover known credits and fail loudly when it cannot."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scholartrack.data.profiling import infer_credits


def test_recovers_known_credits_from_rounded_summary():
    rng = np.random.default_rng(0)
    credits = pd.Series({"A": 3, "B": 3, "C": 2, "D": 1, "E": 1})
    values = pd.DataFrame(rng.uniform(20, 100, size=(150, 5)), columns=credits.index)
    summary = (values * credits).sum(axis=1) / credits.sum()
    result = infer_credits(values, summary.round(1))  # reported to 1 decimal, like the CSV
    assert result.credits.to_dict() == credits.to_dict()
    assert result.max_abs_residual <= 0.05 + 1e-9


def test_ignores_incomplete_rows():
    rng = np.random.default_rng(1)
    credits = pd.Series({"A": 2, "B": 1})
    values = pd.DataFrame(rng.uniform(0, 100, size=(40, 2)), columns=credits.index)
    summary = (values * credits).sum(axis=1) / credits.sum()
    values.iloc[0, 0] = np.nan
    summary.iloc[1] = np.nan
    result = infer_credits(values, summary)
    assert result.n_rows == 38
    assert result.credits.to_dict() == {"A": 2, "B": 1}


def test_too_few_rows():
    values = pd.DataFrame({"A": [1.0], "B": [2.0]})
    with pytest.raises(ValueError, match="Not enough"):
        infer_credits(values, pd.Series([1.5]))
