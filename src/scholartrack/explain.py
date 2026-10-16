"""Explanations for individual predictions and supportive suggestions (Phase 8).

Per-prediction explanation — *feature perturbation* (works for every model):
for each indicator, replace the student's value by the training median (a "typical" value),
re-score, and report how many risk-score points the student's actual value adds or removes.
Missing-value indicators move together with their feature. For the LSTM only the current
(last) semester is perturbed; the earlier history is kept.

This shows what drives *the model's* estimate. It is an association learned from data, not a
cause: "attendance adds +9 points" does not mean attendance caused the risk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scholartrack.features.preprocess import Preprocessor
from scholartrack.features.specs import FEATURE_SPECS
from scholartrack.models.artifacts import Predictor
from scholartrack.paths import CONFIG_DIR
from scholartrack.risk import risk_score

LABELS = {
    "semester": "Semester",
    "prior_sgpa": "Previous semester SGPA",
    "sgpa_current": "Current semester SGPA",
    "sgpa_change": "Change in SGPA",
    "credit_weighted_pct": "Credit-weighted marks",
    "n_failed": "Subjects not passed",
    "share_subjects_below_50": "Share of subjects below 50 %",
    "min_subject_pct": "Weakest subject total",
    "cie_mean_pct": "Average internal assessment (CIE)",
    "ete_mean_pct": "Average end-term exam (ETE)",
    "termwork_mean_pct": "Average termwork",
    "attendance_pct": "Attendance",
    "min_subject_attendance": "Lowest subject attendance",
    "share_subjects_attendance_below_75": "Share of subjects with attendance < 75 %",
    "study_hours_per_week": "Self-reported study hours",
}


# Indicators are explained in concept groups and perturbed together. One-at-a-time perturbation
# of correlated indicators gives confusing signs (with SGPA known, the failure count alone can
# appear to *lower* risk); a group's joint effect is the meaningful quantity.
GROUPS: dict[str, tuple[str, tuple[str, ...]]] = {
    "results": (
        "Current-semester results",
        (
            "sgpa_current",
            "credit_weighted_pct",
            "n_failed",
            "share_subjects_below_50",
            "min_subject_pct",
            "cie_mean_pct",
            "ete_mean_pct",
            "termwork_mean_pct",
        ),
    ),
    "history": ("Previous semester and trend", ("prior_sgpa", "sgpa_change")),
    "attendance": (
        "Attendance",
        ("attendance_pct", "min_subject_attendance", "share_subjects_attendance_below_75"),
    ),
    "study_time": ("Self-reported study time", ("study_hours_per_week",)),
    "stage": ("Stage of the programme", ("semester",)),
}

EXPLANATION_NOTE = (
    "Each factor shows how many risk-score points the student's values in that group add (+) or "
    "remove (−) compared with a typical student, according to the model. These are statistical "
    "associations learned from synthetic data, not causes. For example, among students with the "
    "same grades, lower attendance can appear to lower the estimate (they reached those grades "
    "with less attendance); that is not advice to miss classes, and the attendance suggestion "
    "below follows the college's attendance guideline, not the model."
)


def _group_columns(pre: Preprocessor) -> dict[str, list[int]]:
    names = pre.output_names
    out = {}
    for key, (_, features) in GROUPS.items():
        cols = [names.index(f) for f in features if f in names]
        cols += [names.index(f"{f}_missing") for f in features if f"{f}_missing" in names]
        if cols:
            out[key] = cols
    return out


def typical_vector(pre: Preprocessor) -> np.ndarray:
    """Preprocessed vector of a 'typical' student: training medians, nothing missing."""
    scaled = [(pre.medians[f] - pre.means[f]) / pre.stds[f] for f in pre.features]
    return np.array(scaled + [0.0] * len(pre.indicator_features), dtype=np.float32)


def _value(raw: dict[str, Any], feature: str) -> float | None:
    value = raw.get(feature)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return round(float(value), 2)


def explain(
    predictor: Predictor,
    pre: Preprocessor,
    x_row: np.ndarray,
    raw: dict[str, Any],
    seq: np.ndarray | None = None,
    length: int | None = None,
    min_points: float = 1.0,
) -> list[dict[str, Any]]:
    """Concept-group factors for one prediction, largest absolute effect first."""
    base = typical_vector(pre)
    groups = _group_columns(pre)
    variants = [x_row]
    for cols in groups.values():
        v = x_row.copy()
        v[cols] = base[cols]
        variants.append(v)
    xs = np.stack(variants).astype(np.float32)
    if predictor.input_kind == "sequence":
        assert seq is not None and length is not None
        seqs = np.repeat(seq[None], len(xs), axis=0).astype(np.float32)
        seqs[:, length - 1, :] = xs
        p = predictor.predict_proba(xs, seqs, np.full(len(xs), length, dtype=np.int64))
    else:
        p = predictor.predict_proba(xs)
    scores = 100 * np.clip(p, 0, 1)
    factors = []
    for i, key in enumerate(groups, start=1):
        effect = float(scores[0] - scores[i])
        if abs(effect) < min_points:
            continue
        label, features = GROUPS[key]
        factors.append(
            {
                "group": key,
                "label": label,
                "effect_points": round(effect, 1),
                "direction": "raises risk" if effect > 0 else "lowers risk",
                "indicators": [
                    {
                        "feature": f,
                        "label": LABELS.get(f, f),
                        "value": _value(raw, f),
                        "typical_value": round(pre.medians[f], 2),
                        "unit": FEATURE_SPECS[f].unit,
                    }
                    for f in features
                    if f in pre.features
                ],
            }
        )
    factors.sort(key=lambda f: abs(f["effect_points"]), reverse=True)
    return factors


def load_recommendation_rules(path: Path | str | None = None) -> dict[str, Any]:
    path = Path(path) if path is not None else CONFIG_DIR / "recommendations.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def recommend(raw: dict[str, Any], rules: dict[str, Any]) -> dict[str, Any]:
    """Supportive suggestions triggered by the student's observed indicators."""
    items = []
    for rule in rules["rules"]:
        value = raw.get(rule["feature"])
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        fired = ("below" in rule and value < rule["below"]) or (
            "above" in rule and value > rule["above"]
        )
        if fired:
            items.append(
                {
                    "id": rule["id"],
                    "observation": rule["observation"].format(
                        value=value, abs_value=abs(value), percent=100 * value
                    ),
                    "suggestion": rule["suggestion"],
                }
            )
    return {
        "items": items,
        "message": None if items else rules["no_flag_message"],
        "disclaimer": " ".join(rules["disclaimer"].split()),
    }


def score_row(
    predictor: Predictor,
    x_row: np.ndarray,
    seq: np.ndarray | None = None,
    length: int | None = None,
) -> int:
    if predictor.input_kind == "sequence":
        p = predictor.predict_proba(x_row[None], seq[None], np.array([length]))
    else:
        p = predictor.predict_proba(x_row[None])
    return int(risk_score(p)[0])
