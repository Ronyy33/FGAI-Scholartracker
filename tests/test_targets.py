"""The target rule, checked on hand-computed cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scholartrack.targets import (
    TargetConfig,
    load_target_config,
    next_semester_labels,
    semester_outcomes,
    subject_outcomes,
)

CONFIG = TargetConfig("t", horizon_semesters=1, sgpa_threshold=6.0, fail_counts_as_difficulty=True)


def _subject(key, sem, code, total, max_marks=100, ete=None, ete_max=None, credits=3):
    return dict(
        student_key=key,
        semester=sem,
        subject_code=code,
        credits=credits,
        max_marks=max_marks,
        total=total,
        ete=ete,
        ete_max=ete_max,
    )


NUMERIC = ["credits", "max_marks", "total", "ete", "ete_max"]


def _frame(rows):
    """Build a subjects table; None becomes NaN in the numeric columns."""
    frame = pd.DataFrame(rows)
    frame[NUMERIC] = frame[NUMERIC].astype(float)
    return frame


@pytest.mark.parametrize(
    ("total", "ete", "passed"),
    [
        (65, 40, 1.0),  # comfortable pass
        (40, 24, 1.0),  # exactly 40 % overall and 40 % ETE
        (39, 30, 0.0),  # total below 40 %
        (55, 20, 0.0),  # total fine but ETE 20/60 = 33 % < 40 %
        (55, None, np.nan),  # ETE unknown -> cannot decide
        (30, None, 0.0),  # total already fails -> decided without the ETE
        (None, 40, np.nan),
    ],
)
def test_pass_rule(curriculum, total, ete, passed):
    frame = _frame([_subject("S1", 5, "X", total, ete=ete, ete_max=60)])
    result = subject_outcomes(frame, curriculum.grading)["passed"].iloc[0]
    assert (np.isnan(result) and np.isnan(passed)) or result == passed


def test_termwork_subject_has_no_ete_requirement(curriculum):
    frame = _frame([_subject("S1", 5, "TW", 12, max_marks=25)])
    assert subject_outcomes(frame, curriculum.grading)["passed"].iloc[0] == 1.0  # 48 %


def test_sgpa_and_difficulty(curriculum):
    rows = [
        _subject("S1", 1, "A", 85, ete=50, ete_max=60, credits=3),  # 9 points
        _subject("S1", 1, "B", 72, ete=45, ete_max=60, credits=1),  # 8 points
        _subject("S2", 1, "A", 55, ete=33, ete_max=60, credits=3),  # 6 points
        _subject("S2", 1, "B", 35, ete=20, ete_max=60, credits=1),  # fail -> 0
    ]
    results = subject_outcomes(_frame(rows), curriculum.grading)
    out = semester_outcomes(results, CONFIG).set_index("student_key")
    assert out.at["S1", "sgpa"] == pytest.approx((9 * 3 + 8 * 1) / 4)
    assert out.at["S1", "difficulty"] == 0.0
    assert out.at["S2", "sgpa"] == pytest.approx((6 * 3 + 0) / 4)
    assert out.at["S2", "n_failed"] == 1
    assert out.at["S2", "difficulty"] == 1.0


def test_low_sgpa_without_failure_is_difficulty(curriculum):
    rows = [_subject("S1", 1, "A", 52, ete=33, ete_max=60)]  # pass, 5 points < 6.0
    out = semester_outcomes(subject_outcomes(_frame(rows), curriculum.grading), CONFIG)
    assert out["n_failed"].item() == 0 and out["difficulty"].item() == 1.0


def test_unknown_outcome_stays_unknown_unless_a_failure_is_known(curriculum):
    rows = [
        _subject("S1", 1, "A", 80, ete=None, ete_max=60),  # unknown
        _subject("S1", 1, "B", 80, ete=50, ete_max=60),
        _subject("S2", 1, "A", 80, ete=None, ete_max=60),  # unknown
        _subject("S2", 1, "B", 20, ete=10, ete_max=60),  # known failure
    ]
    out = semester_outcomes(subject_outcomes(_frame(rows), curriculum.grading), CONFIG)
    out = out.set_index("student_key")
    assert np.isnan(out.at["S1", "difficulty"]) and np.isnan(out.at["S1", "sgpa"])
    assert out.at["S2", "difficulty"] == 1.0


def test_next_semester_labels_use_only_the_future():
    outcomes = pd.DataFrame(
        {
            "student_key": ["S1", "S1", "S1", "S2", "S2"],
            "semester": [1, 2, 3, 1, 3],  # S2 has no semester-2 record
            "difficulty": [0.0, 1.0, np.nan, 1.0, 0.0],
        }
    )
    labels = next_semester_labels(outcomes, CONFIG).set_index(["student_key", "semester"])
    assert labels.at[("S1", 1), "label"] == 1.0  # from semester 2
    assert np.isnan(labels.at[("S1", 2), "label"])  # semester 3 outcome undetermined
    assert labels.at[("S1", 2), "label_status"] == "future_outcome_undetermined"
    assert labels.at[("S1", 3), "label_status"] == "future_semester_not_observed"
    assert labels.at[("S2", 1), "label_status"] == "future_semester_not_observed"


def test_config_file_loads():
    config = load_target_config()
    assert config.horizon_semesters == 1 and 0 < config.sgpa_threshold <= 10
