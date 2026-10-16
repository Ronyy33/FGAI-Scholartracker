"""Feature metadata and configuration.

``FEATURE_SPECS`` is the single description of every engineered feature; the data dictionary
(``docs/DATA_DICTIONARY.md``) is generated from it so documentation and code cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from scholartrack.paths import CONFIG_DIR


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    description: str
    dtype: str
    unit: str
    value_range: str
    available: str  # when the value is known
    source: str  # which raw fields it comes from


FEATURE_SPECS: dict[str, FeatureSpec] = {
    s.name: s
    for s in [
        FeatureSpec(
            "semester",
            "Semester number t of the record",
            "integer",
            "semester",
            "1–8",
            "start of semester t",
            "semester",
        ),
        FeatureSpec(
            "prior_sgpa",
            "SGPA of semester t − 1 as reported in the record",
            "float",
            "SGPA",
            "0–10 (missing in semester 1)",
            "start of semester t",
            "prior_sgpa",
        ),
        FeatureSpec(
            "sgpa_current",
            "SGPA of semester t computed from subject totals with the grading rules "
            "(credit-weighted grade points over subjects whose result is known)",
            "float",
            "SGPA",
            "0–10",
            "end of semester t",
            "subject totals, ETE marks, credits",
        ),
        FeatureSpec(
            "sgpa_change",
            "sgpa_current − prior_sgpa (momentum)",
            "float",
            "SGPA",
            "−10–10",
            "end of semester t",
            "sgpa_current, prior_sgpa",
        ),
        FeatureSpec(
            "credit_weighted_pct",
            "Credit-weighted mean of subject total percentages in semester t",
            "float",
            "%",
            "0–100",
            "end of semester t",
            "subject totals, max marks, credits",
        ),
        FeatureSpec(
            "n_failed",
            "Number of subjects failed in semester t (pass rule in configs/curriculum.yaml)",
            "integer",
            "subjects",
            "0–11",
            "end of semester t",
            "subject totals, ETE marks",
        ),
        FeatureSpec(
            "share_subjects_below_50",
            "Share of semester-t subjects whose total is below 50 %",
            "float",
            "fraction",
            "0–1",
            "end of semester t",
            "subject totals",
        ),
        FeatureSpec(
            "min_subject_pct",
            "Lowest subject total percentage in semester t",
            "float",
            "%",
            "0–100",
            "end of semester t",
            "subject totals",
        ),
        FeatureSpec(
            "cie_mean_pct",
            "Mean CIE percentage over theory subjects in semester t",
            "float",
            "%",
            "0–100",
            "during semester t",
            "CIE marks",
        ),
        FeatureSpec(
            "ete_mean_pct",
            "Mean ETE percentage over subjects with an end-term exam in semester t",
            "float",
            "%",
            "0–100",
            "end of semester t",
            "ETE marks",
        ),
        FeatureSpec(
            "termwork_mean_pct",
            "Mean termwork percentage over termwork subjects in semester t",
            "float",
            "%",
            "0–100",
            "end of semester t",
            "termwork marks",
        ),
        FeatureSpec(
            "attendance_pct",
            "Credit-weighted mean subject attendance in semester t (recomputed)",
            "float",
            "%",
            "0–100",
            "during semester t",
            "subject attendance, credits",
        ),
        FeatureSpec(
            "min_subject_attendance",
            "Lowest subject attendance in semester t",
            "float",
            "%",
            "0–100",
            "during semester t",
            "subject attendance",
        ),
        FeatureSpec(
            "share_subjects_attendance_below_75",
            "Share of semester-t subjects with attendance below 75 %",
            "float",
            "fraction",
            "0–1",
            "during semester t",
            "subject attendance",
        ),
        FeatureSpec(
            "study_hours_per_week",
            "Self-reported weekly study hours in semester t",
            "float",
            "hours/week",
            "0–100",
            "during semester t (self-reported)",
            "study_hours_per_week",
        ),
    ]
}


@dataclass(frozen=True)
class FeatureConfig:
    snapshot_features: tuple[str, ...]
    missing_indicators: tuple[str, ...]
    low_marks_pct: float
    low_attendance_pct: float
    excluded: dict[str, str]
    split_seed: int
    split_fractions: dict[str, float]
    max_sequence_length: int
    raw: dict[str, Any]


def load_feature_config(path: Path | str | None = None) -> FeatureConfig:
    path = Path(path) if path is not None else CONFIG_DIR / "features.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    snapshot = tuple(raw["snapshot_features"])
    unknown = [f for f in snapshot if f not in FEATURE_SPECS]
    if unknown:
        raise ValueError(f"Unknown features in {path.name}: {unknown}")
    indicators = tuple(raw["missing_indicators"])
    if not set(indicators) <= set(snapshot):
        raise ValueError("missing_indicators must be a subset of snapshot_features")
    overlap = set(snapshot) & set(raw["excluded"])
    if overlap:
        raise ValueError(f"Features both used and excluded: {sorted(overlap)}")
    fractions = {k: float(v) for k, v in raw["split"]["fractions"].items()}
    if abs(sum(fractions.values()) - 1) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    return FeatureConfig(
        snapshot_features=snapshot,
        missing_indicators=indicators,
        low_marks_pct=float(raw["thresholds"]["low_marks_pct"]),
        low_attendance_pct=float(raw["thresholds"]["low_attendance_pct"]),
        excluded=dict(raw["excluded"]),
        split_seed=int(raw["split"]["seed"]),
        split_fractions=fractions,
        max_sequence_length=int(raw["sequence"]["max_length"]),
        raw=raw,
    )
