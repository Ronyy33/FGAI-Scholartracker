"""Academic outcomes and the "falling behind" label.

This module is the single place where the target is defined. It works on the adapter's
``subjects`` table, so the same rule labels the reference CSV, the synthetic data and any
future dataset.

Three-valued logic is used on purpose: an outcome is 1 (yes), 0 (no) or NaN (cannot be
determined, e.g. an ETE mark is missing). Unknown outcomes are never silently counted as
"not at risk".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scholartrack.data.curriculum import GradingRules
from scholartrack.paths import CONFIG_DIR


@dataclass(frozen=True)
class TargetConfig:
    name: str
    horizon_semesters: int
    sgpa_threshold: float
    fail_counts_as_difficulty: bool


def load_target_config(path: Path | str | None = None) -> TargetConfig:
    path = Path(path) if path is not None else CONFIG_DIR / "target.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TargetConfig(
        name=str(raw["name"]),
        horizon_semesters=int(raw["horizon_semesters"]),
        sgpa_threshold=float(raw["sgpa_threshold"]),
        fail_counts_as_difficulty=bool(raw["fail_counts_as_difficulty"]),
    )


def subject_outcomes(subjects: pd.DataFrame, grading: GradingRules) -> pd.DataFrame:
    """Add ``passed`` (1/0/NaN) and ``grade_points`` to an adapter ``subjects`` table."""
    out = subjects.copy()
    total_pct = out["total"] / out["max_marks"] * 100
    ete_pct = out["ete"] / out["ete_max"] * 100
    has_ete = out["ete_max"].notna()

    total_ok = total_pct >= grading.pass_total_pct
    ete_ok = ete_pct >= grading.pass_ete_pct
    passed = np.where(
        total_pct.isna(),
        np.nan,
        np.where(
            ~total_ok,
            0.0,  # failing the total fails the subject, whatever the ETE says
            np.where(~has_ete, 1.0, np.where(ete_pct.isna(), np.nan, ete_ok.astype(float))),
        ),
    )
    out["passed"] = passed
    points = grading.points_for(total_pct.to_numpy(), np.nan_to_num(passed, nan=1.0) == 1.0)
    out["grade_points"] = np.where(np.isnan(passed), np.nan, points)
    return out


def semester_outcomes(subject_results: pd.DataFrame, config: TargetConfig) -> pd.DataFrame:
    """Per (student_key, semester): SGPA, failed-subject count and the difficulty flag."""
    frame = subject_results.copy()
    frame["weighted_points"] = frame["grade_points"] * frame["credits"]
    frame["failed"] = (frame["passed"] == 0).astype(float)
    frame["unknown"] = frame["passed"].isna() | frame["credits"].isna()

    grouped = frame.groupby(["student_key", "semester"], sort=True)
    result = pd.DataFrame(
        {
            "n_subjects": grouped.size(),
            "n_failed": grouped["failed"].sum(),
            "n_unknown": grouped["unknown"].sum(),
            "credits": grouped["credits"].sum(),
            "weighted_points": grouped["weighted_points"].sum(),
        }
    )
    complete = result["n_unknown"] == 0
    result["sgpa"] = np.where(complete, result["weighted_points"] / result["credits"], np.nan)
    result["sgpa"] = result["sgpa"].round(2)

    any_failed = result["n_failed"] > 0
    low_sgpa = result["sgpa"] < config.sgpa_threshold
    fail_rule = any_failed if config.fail_counts_as_difficulty else pd.Series(False, result.index)
    # 1 if a known failure or a known low SGPA; 0 only if everything is known and fine.
    result["difficulty"] = np.where(fail_rule | low_sgpa, 1.0, np.where(complete, 0.0, np.nan))
    return result.drop(columns=["weighted_points"]).reset_index()


def next_semester_labels(outcomes: pd.DataFrame, config: TargetConfig) -> pd.DataFrame:
    """Attach the label for semester t + horizon to each (student_key, semester t) row.

    Rows whose future semester is not observed (latest semester, or the student left) get
    ``label = NaN`` and ``label_status`` explaining why; they must not be used for training.
    """
    h = config.horizon_semesters
    future = outcomes[["student_key", "semester", "difficulty"]].copy()
    future["semester"] = future["semester"] - h
    merged = outcomes[["student_key", "semester"]].merge(
        future.rename(columns={"difficulty": "label"}), on=["student_key", "semester"], how="left"
    )
    observed = (
        merged.merge(
            future[["student_key", "semester"]].assign(_seen=True),
            on=["student_key", "semester"],
            how="left",
        )["_seen"]
        .fillna(False)
        .astype(bool)
    )
    merged["label_status"] = np.select(
        [~observed, merged["label"].isna()],
        ["future_semester_not_observed", "future_outcome_undetermined"],
        default="labelled",
    )
    return merged
