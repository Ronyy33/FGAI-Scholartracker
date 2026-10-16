"""Risk score, calibration and threshold metrics on hand-computed cases."""

from __future__ import annotations

import math

import numpy as np
import pytest
import yaml

from scholartrack.evaluation.calibration import (
    PlattCalibrator,
    apply_calibration,
    choose_calibration,
    expected_calibration_error,
    reliability_table,
)
from scholartrack.evaluation.classification import (
    choose_threshold,
    student_bootstrap,
    threshold_metrics,
    top_k_capture,
)
from scholartrack.risk import load_risk_bands, risk_band, risk_score


@pytest.mark.parametrize(
    ("p", "score", "band"),
    [
        (0.0, 0, "Low"),
        (0.18, 18, "Low"),
        (0.294, 29, "Low"),
        (0.295, 30, "Moderate"),
        (0.54, 54, "Moderate"),
        (0.599, 60, "High"),
        (0.86, 86, "High"),
        (1.0, 100, "High"),
    ],
)
def test_risk_score_and_band_boundaries(p, score, band):
    bands = load_risk_bands()
    s = risk_score(np.array([p]))
    assert s[0] == score
    assert risk_band(s, bands)[0] == band


def test_risk_bands_must_cover_0_to_100(tmp_path):
    path = tmp_path / "bands.yaml"
    path.write_text(
        yaml.safe_dump(
            {"bands": [{"name": "A", "min": 0, "max": 49}, {"name": "B", "min": 51, "max": 100}]}
        )
    )
    with pytest.raises(ValueError, match="without gaps"):
        load_risk_bands(path)


def test_threshold_metrics_hand_computed():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.6, 0.2, 0.7, 0.3, 0.1, 0.1, 0.05])
    m = threshold_metrics(y, p, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 4)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["specificity"] == pytest.approx(4 / 5)


def test_undefined_precision_is_nan_not_zero():
    m = threshold_metrics(np.array([0, 1]), np.array([0.1, 0.2]), 0.9)
    assert m["tp"] + m["fp"] == 0 and math.isnan(m["precision"]) and math.isnan(m["f1"])


def test_choose_threshold_maximises_f1():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    thr = choose_threshold(y, p)
    assert thr == pytest.approx(0.35)  # flags 0.35, 0.4, 0.8 -> F1 0.8 (best possible here)


def test_top_k_capture():
    y = np.array([1, 0, 1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    c = top_k_capture(y, p, k=2)
    assert c == {
        "k": 2,
        "positives_in_top_k": 1,
        "total_positives": 3,
        "capture_pct": pytest.approx(100 / 3),
        "precision_at_k_pct": 50.0,
    }


def test_perfectly_calibrated_has_small_ece_and_platt_fixes_miscalibration():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.99, 20000)
    y = (rng.random(20000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.02
    overconfident = 1 / (1 + np.exp(-3 * np.log(p / (1 - p))))  # logits stretched x3
    assert expected_calibration_error(y, overconfident) > 0.05
    fixed = PlattCalibrator.fit(overconfident, y).transform(overconfident)
    assert expected_calibration_error(y, fixed) < 0.02
    table = reliability_table(y, p)
    assert table["count"].sum() == 20000


def test_choose_calibration_keeps_good_models_and_fixes_bad_ones():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.01, 0.99, 4000)
    y = (rng.random(4000) < p).astype(int)
    groups = np.repeat(np.arange(1000), 4)
    assert choose_calibration(y, p, groups, 5, 0.001)["method"] == "none"
    bad = 1 / (1 + np.exp(-3 * np.log(p / (1 - p))))
    choice = choose_calibration(y, bad, groups, 5, 0.001)
    assert choice["method"] == "platt"
    assert expected_calibration_error(y, apply_calibration(bad, choice)) < 0.03


def test_student_bootstrap_resamples_whole_students():
    groups = np.array(["a", "a", "b", "b", "c"])
    seen = []

    def stat(idx):
        seen.append(idx)
        return {"n": float(len(idx))}

    ci = student_bootstrap(groups, stat, n_resamples=50, seed=0)
    for idx in seen:  # rows 0-1 and 2-3 always appear together
        assert list(idx).count(0) == list(idx).count(1)
        assert list(idx).count(2) == list(idx).count(3)
    assert ci["n"][0] <= ci["n"][1]


def test_constant_predictions_get_a_threshold_that_flags_everyone():
    y = np.array([0, 1, 0, 0, 1])
    p = np.full(5, 0.20684617)  # e.g. the majority baseline
    thr = choose_threshold(y, p)
    assert threshold_metrics(y, p, thr)["flagged_share"] == 1.0
