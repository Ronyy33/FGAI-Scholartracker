"""Synthetic demonstration population for the dashboard.

The synthetic dataset is regenerated deterministically (``generate_synthetic_data.py``) and must
have the exact manifest hash recorded in the serving bundle, so the demo shows the same data the
models were evaluated on. "Current students" are those whose latest record is in the latest
academic year and who have a next semester in the programme — the population a college would
actually score today. Analytics suppress groups smaller than ``MIN_GROUP`` records.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scholartrack.api.service import ScoringService
from scholartrack.features.build import semester_features
from scholartrack.training.data import load_dataset

MIN_GROUP = 10


class DemoUnavailable(RuntimeError):
    pass


class DemoData:
    def __init__(self, data_dir: Path, service: ScoringService) -> None:
        try:
            semesters, subjects, sha = load_dataset(data_dir)
        except (FileNotFoundError, ValueError) as exc:
            raise DemoUnavailable(str(exc)) from exc
        if sha != service.bundle["dataset_manifest_sha256"]:
            raise DemoUnavailable(
                "Synthetic dataset differs from the one the models were "
                "evaluated on; regenerate it with the committed config."
            )
        self.service = service
        self.semesters, self.subjects = semesters, subjects
        students = pd.read_csv(data_dir / "students.csv")
        self.prn_of = dict(zip(students["student_key"], students["prn"], strict=True))
        self.key_of = {v: k for k, v in self.prn_of.items()}
        self.cohort_of = dict(zip(students["student_key"], students["cohort"], strict=True))
        self.features = semester_features(semesters, subjects, service.curriculum, service.features)

        latest = semesters.loc[semesters.groupby("student_key")["semester"].idxmax()]
        current_year = semesters["academic_year"].max()
        current = latest[(latest["academic_year"] == current_year) & (latest["semester"] < 8)]
        keys = set(current["student_key"])
        scored = service.score(
            semesters[semesters["student_key"].isin(keys)],
            subjects[subjects["student_key"].isin(keys)],
            with_explanations=False,
        )
        self.current = pd.DataFrame(
            [
                {
                    "prn": self.prn_of[r["student_key"]],
                    "student_key": r["student_key"],
                    "cohort": int(self.cohort_of[r["student_key"]]),
                    "semester": r["semester"],
                    "risk_score": r["risk_score"],
                    "risk_band": r["risk_band"],
                    "flagged": r["flagged"],
                    "sgpa_current": r["indicators"]["sgpa_current"],
                    "attendance_pct": r["indicators"]["attendance_pct"],
                    "n_failed": r["indicators"]["n_failed"],
                }
                for r in scored
            ]
        ).sort_values("risk_score", ascending=False, ignore_index=True)
        self.current_year = current_year

    def overview(self) -> dict[str, Any]:
        bands = self.current["risk_band"].value_counts()
        n = len(self.current)
        return {
            "synthetic_data": True,
            "records_processed": int(len(self.semesters)),
            "students": int(self.semesters["student_key"].nunique()),
            "current_academic_year": self.current_year,
            "current_students_scored": n,
            "risk_bands": [
                {
                    "band": b.name,
                    "range": [b.minimum, b.maximum],
                    "count": int(bands.get(b.name, 0)),
                    "percent": round(100 * bands.get(b.name, 0) / n, 1) if n else 0.0,
                }
                for b in self.service.bands
            ],
            "flagged": int(self.current["flagged"].sum()),
        }

    def list_students(
        self, query: str = "", band: str | None = None, limit: int = 50, offset: int = 0
    ) -> dict[str, Any]:
        frame = self.current
        if query:
            frame = frame[frame["prn"].str.contains(query.strip().upper(), regex=False)]
        if band:
            frame = frame[frame["risk_band"] == band]
        page = frame.iloc[offset : offset + limit].drop(columns="student_key")
        return {"total": int(len(frame)), "items": _records(page)}

    def student_detail(self, prn: str) -> dict[str, Any] | None:
        key = self.key_of.get(prn.strip().upper())
        if key is None:
            return None
        history = self.features[self.features["student_key"] == key].sort_values("semester")
        sem = self.semesters[self.semesters["student_key"] == key]
        years = dict(zip(sem["semester"], sem["academic_year"], strict=True))
        timeline = [
            {
                "semester": int(h["semester"]),
                "academic_year": years[h["semester"]],
                "sgpa": _num(h["sgpa_current"]),
                "credit_weighted_pct": _num(h["credit_weighted_pct"]),
                "attendance_pct": _num(h["attendance_pct"]),
                "n_failed": _num(h["n_failed"]),
            }
            for _, h in history.iterrows()
        ]
        is_current = key in set(self.current["student_key"])
        prediction = None
        if is_current:
            prediction = self.service.score(
                sem, self.subjects[self.subjects["student_key"] == key]
            )[0]
            prediction.pop("student_key")
        return {
            "prn": prn.upper(),
            "cohort": int(self.cohort_of[key]),
            "history": timeline,
            "prediction": prediction,
            "message": None
            if is_current
            else "This synthetic student is not enrolled in the current academic year (graduated "
            "or left), so there is no next semester to predict.",
        }

    def analytics(self, cohort: int | None = None, semester: int | None = None) -> dict[str, Any]:
        feats = self.features.copy()
        feats["cohort"] = feats["student_key"].map(self.cohort_of)
        if cohort is not None:
            feats = feats[feats["cohort"] == cohort]
        current = self.current if cohort is None else self.current[self.current["cohort"] == cohort]

        bins = list(range(0, 101, 10))
        hist = pd.cut(current["risk_score"], bins=bins, right=False, include_lowest=True)
        distribution = [
            {"from": int(b.left), "to": int(min(b.right, 100)), "count": int(c)}
            for b, c in hist.value_counts(sort=False).items()
        ]

        view = feats if semester is None else feats[feats["semester"] == semester]
        grid = (
            view.assign(
                att=(view["attendance_pct"] // 5 * 5).clip(upper=95),
                marks=(view["credit_weighted_pct"] // 5 * 5).clip(upper=95),
            )
            .groupby(["att", "marks"])
            .size()
            .reset_index(name="count")
        )
        suppressed = int((grid["count"] < MIN_GROUP).sum())
        grid = grid[grid["count"] >= MIN_GROUP]

        trends = feats.groupby(["cohort", "semester"])["sgpa_current"].agg(["mean", "size"])
        trends = trends[trends["size"] >= MIN_GROUP].reset_index()

        sem_for_subjects = semester or 5
        subj = self.subjects[self.subjects["semester"] == sem_for_subjects]
        if cohort is not None:
            subj = subj[subj["student_key"].map(self.cohort_of) == cohort]
        by_subject = (
            subj.groupby("subject_code")
            .agg(
                n=("total_pct", "size"),
                mean_total_pct=("total_pct", "mean"),
                below_40_pct=("total_pct", lambda s: 100 * (s < 40).mean()),
                mean_attendance=("attendance_pct", "mean"),
            )
            .reset_index()
        )
        by_subject = by_subject[by_subject["n"] >= MIN_GROUP]

        return {
            "synthetic_data": True,
            "filters": {
                "cohort": cohort,
                "semester": semester,
                "cohorts": sorted(int(c) for c in set(self.cohort_of.values())),
                "semesters": list(range(1, 9)),
            },
            "min_group_size": MIN_GROUP,
            "risk_distribution": distribution,
            "attendance_vs_marks": {
                "cells": _records(
                    grid.rename(columns={"att": "attendance_from", "marks": "marks_from"})
                ),
                "suppressed_cells": suppressed,
            },
            "sgpa_trends": _records(trends.rename(columns={"mean": "mean_sgpa", "size": "n"})),
            "subjects": {"semester": sem_for_subjects, "items": _records(by_subject)},
        }


def _num(value: Any) -> float | None:
    return (
        None
        if value is None or (isinstance(value, float) and np.isnan(value))
        else round(float(value), 2)
    )


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for record in frame.to_dict("records"):
        out.append(
            {
                k: (
                    _num(v)
                    if isinstance(v, float)
                    else int(v)
                    if isinstance(v, np.integer)
                    else bool(v)
                    if isinstance(v, np.bool_)
                    else v
                )
                for k, v in record.items()
            }
        )
    return out
