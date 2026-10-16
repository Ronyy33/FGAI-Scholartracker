"""Feature engineering on the adapter's tables.

Input: the ``semesters`` and ``subjects`` tables produced by ``adapt_college_frame`` (or the
synthetic generator, which emits the same columns). Output: one row per (student_key, semester)
with *snapshot* features that use **only that semester's records plus the reported
``prior_sgpa``**. Because nothing else is used, a single row of a college-format CSV is enough
to compute them — which is exactly what the batch-prediction page receives.

The modelling table adds the next-semester label (``scholartrack.targets``) to each row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scholartrack.data.curriculum import Curriculum
from scholartrack.features.specs import FeatureConfig
from scholartrack.targets import (
    TargetConfig,
    next_semester_labels,
    semester_outcomes,
    subject_outcomes,
)

KEYS = ["student_key", "semester"]


def _weighted_mean(values: pd.Series, weights: pd.Series, groups: list[pd.Series]) -> pd.Series:
    """Credit-weighted mean per group over rows where both value and weight are known."""
    known = values.notna() & weights.notna()
    num = (values.where(known) * weights.where(known)).groupby(groups).sum(min_count=1)
    den = weights.where(known).groupby(groups).sum(min_count=1)
    return num / den


def semester_features(
    semesters: pd.DataFrame, subjects: pd.DataFrame, curriculum: Curriculum, config: FeatureConfig
) -> pd.DataFrame:
    """Snapshot features for every (student_key, semester) in ``semesters``."""
    subj = subject_outcomes(subjects, curriculum.grading)
    subj["cie_pct"] = subj["cie"] / subj["cie_max"] * 100
    subj["ete_pct"] = subj["ete"] / subj["ete_max"] * 100
    subj["termwork_pct"] = subj["termwork"] / subj["max_marks"] * 100
    groups = [subj["student_key"], subj["semester"]]
    grouped = subj.groupby(groups)

    has_total = subj["total_pct"].notna()
    has_att = subj["attendance_pct"].notna()
    feats = pd.DataFrame(
        {
            "sgpa_current": _weighted_mean(subj["grade_points"], subj["credits"], groups),
            "credit_weighted_pct": _weighted_mean(subj["total_pct"], subj["credits"], groups),
            "n_failed": (subj["passed"] == 0).groupby(groups).sum(),
            "share_subjects_below_50": (subj["total_pct"] < config.low_marks_pct)
            .where(has_total)
            .groupby(groups)
            .mean(),
            "min_subject_pct": grouped["total_pct"].min(),
            "cie_mean_pct": grouped["cie_pct"].mean(),
            "ete_mean_pct": grouped["ete_pct"].mean(),
            "termwork_mean_pct": grouped["termwork_pct"].mean(),
            "attendance_pct": _weighted_mean(subj["attendance_pct"], subj["credits"], groups),
            "min_subject_attendance": grouped["attendance_pct"].min(),
            "share_subjects_attendance_below_75": (
                subj["attendance_pct"] < config.low_attendance_pct
            )
            .where(has_att)
            .groupby(groups)
            .mean(),
        }
    )
    feats.index.names = KEYS
    feats = feats.reset_index()

    base = semesters[[*KEYS, "prior_sgpa", "study_hours_per_week"]].copy()
    out = base.merge(feats, on=KEYS, how="left")
    out["sgpa_current"] = out["sgpa_current"].round(4)
    out["sgpa_change"] = out["sgpa_current"] - out["prior_sgpa"]
    for column in out.columns.difference(KEYS):
        out[column] = out[column].astype(float)
    return out.sort_values(KEYS, ignore_index=True)


def build_modeling_table(
    semesters: pd.DataFrame,
    subjects: pd.DataFrame,
    curriculum: Curriculum,
    target: TargetConfig,
    config: FeatureConfig,
) -> pd.DataFrame:
    """Snapshot features + next-semester label + entry year, one row per student-semester."""
    feats = semester_features(semesters, subjects, curriculum, config)
    outcomes = semester_outcomes(subject_outcomes(subjects, curriculum.grading), target)
    labels = next_semester_labels(outcomes, target)
    table = feats.merge(labels, on=KEYS, how="left")
    table["label_status"] = table["label_status"].fillna("future_semester_not_observed")
    table["entry_year"] = entry_year(semesters).reindex(table["student_key"]).to_numpy()
    return table[["student_key", "entry_year", *config.snapshot_features, "label", "label_status"]]


def entry_year(semesters: pd.DataFrame) -> pd.Series:
    """Admission year per student, derived from academic year and semester (no identifiers).

    Semester s of a student admitted in year Y falls in the academic year starting
    Y + (s - 1) // 2, so Y = start year - (s - 1) // 2.
    """
    start = semesters["academic_year"].str.slice(0, 4).astype(float)
    year = start - (semesters["semester"] - 1) // 2
    return year.groupby(semesters["student_key"]).min().astype("Int64")


def build_sequences(
    table: pd.DataFrame, matrix: np.ndarray, max_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Turn per-row feature vectors into per-row histories for the LSTM.

    ``table`` holds ``student_key`` and ``semester`` for each row of ``matrix`` (already
    preprocessed). For the row (student, t) the sequence is that student's vectors for their
    observed semesters <= t, in time order, left-aligned and zero-padded to ``max_length``.
    Returns ``(sequences [n, max_length, f], lengths [n])``.
    """
    n, f = matrix.shape
    sequences = np.zeros((n, max_length, f), dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int64)
    order = np.lexsort((table["semester"].to_numpy(), table["student_key"].to_numpy()))
    keys = table["student_key"].to_numpy()[order]
    boundaries = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1], True])
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        rows = order[start:stop]
        history = matrix[rows]
        for i, row in enumerate(rows):
            prefix = history[max(0, i + 1 - max_length) : i + 1]
            sequences[row, : len(prefix)] = prefix
            lengths[row] = len(prefix)
    return sequences, lengths
