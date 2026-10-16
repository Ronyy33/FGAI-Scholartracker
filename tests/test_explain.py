"""Explanations and supportive suggestions."""

from __future__ import annotations

import numpy as np
import pytest

from scholartrack.explain import explain, load_recommendation_rules, recommend, typical_vector
from scholartrack.features.preprocess import Preprocessor
from scholartrack.models.artifacts import Predictor
from scholartrack.models.baselines import LogisticModel

FEATURES = ("attendance_pct", "n_failed", "prior_sgpa")


@pytest.fixture()
def setup():
    import pandas as pd

    train = pd.DataFrame(
        {
            "attendance_pct": [70.0, 80.0, 90.0],
            "n_failed": [0.0, 0.0, 2.0],
            "prior_sgpa": [6.0, 7.0, np.nan],
        }
    )
    pre = Preprocessor.fit(train, FEATURES, ("prior_sgpa",))
    # Hand-set model: low attendance and failures raise risk; prior SGPA is ignored.
    model = LogisticModel(tuple(pre.output_names), (-1.0, 1.5, 0.0, 0.0), -1.0, 1.0, None)
    return pre, Predictor("logistic_regression", "snapshot", model)


def test_typical_student_has_no_factors(setup):
    pre, predictor = setup
    assert explain(predictor, pre, typical_vector(pre), {}) == []


def test_group_factors_follow_the_model(setup):
    pre, predictor = setup
    import pandas as pd

    raw = {"attendance_pct": 60.0, "n_failed": 2.0, "prior_sgpa": 7.0}
    x = pre.transform(pd.DataFrame([raw]))[0]
    factors = {f["group"]: f for f in explain(predictor, pre, x, raw)}
    assert factors["attendance"]["direction"] == "raises risk"  # coefficient -1, below median
    assert factors["results"]["direction"] == "raises risk"  # failures, coefficient +1.5
    assert "history" not in factors  # prior SGPA has zero coefficient -> no effect
    att = factors["attendance"]["indicators"][0]
    assert (att["value"], att["typical_value"]) == (60.0, 80.0)


def test_failures_and_grades_are_explained_together(setup):
    """A failed subject must never be reported as lowering risk on its own."""
    pre, predictor = setup
    groups = {f["group"] for f in explain(predictor, pre, typical_vector(pre), {})}
    assert groups == set()
    from scholartrack.explain import GROUPS

    assert {"sgpa_current", "n_failed"} <= set(GROUPS["results"][1])


def test_recommendations_fire_on_indicators_and_stay_supportive():
    rules = load_recommendation_rules()
    out = recommend({"attendance_pct": 62.0, "n_failed": 1.0, "sgpa_change": -1.2}, rules)
    ids = [i["id"] for i in out["items"]]
    assert ids == ["attendance", "failed_subject", "declining_sgpa"]
    assert "62 %" in out["items"][0]["observation"]
    text = " ".join(i["suggestion"] for i in out["items"]).lower()
    for word in ("penal", "punish", "expel", "suspend", "lazy", "warning letter"):
        assert word not in text
    assert "decision support" in out["disclaimer"]


def test_no_flags_gives_neutral_message():
    out = recommend({"attendance_pct": 92.0, "n_failed": 0.0}, load_recommendation_rules())
    assert out["items"] == [] and "regular mentoring" in out["message"]
