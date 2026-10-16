"""Adapter: college-format CSV -> ScholarTrack's common internal representation.

Different sources (the reference CSV, the synthetic generator, a future public dataset) are
mapped to the same two tables so every later step works the same way:

``semesters``  one row per (student_key, semester) with student-level fields;
``subjects``   one row per (student_key, semester, subject_code) with marks and attendance.

Rules applied here:

1. Direct identifiers (PRN, name) never enter these tables. Each student gets an opaque
   ``student_key`` (S000001, ...). The key -> PRN map is returned separately, kept in memory,
   and is only used to label predictions for the person who uploaded the file.
2. Rows with validation *errors* are dropped and listed.
3. Cells flagged *invalid* become missing (NaN). Nothing is "corrected" by guessing.
4. Columns that do not belong to a subject's assessment scheme are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scholartrack.data.curriculum import SCHEME_COMPONENTS, Curriculum
from scholartrack.data.schema import STUDENT_FIELDS, subject_columns
from scholartrack.data.validation import (
    ValidationReport,
    ValidationResult,
    infer_subject_spec,
    validate_college_frame,
)

SEMESTER_COLUMNS = [f.name for f in STUDENT_FIELDS if f.role != "identifier"]
SUBJECT_TABLE_COLUMNS = [
    "student_key",
    "semester",
    "subject_code",
    "scheme",
    "credits",
    "max_marks",
    "cie_max",
    "ete_max",
    "attendance_pct",
    "cie",
    "ete",
    "termwork",
    "total",
    "total_pct",
]


@dataclass
class AdaptedData:
    semesters: pd.DataFrame
    subjects: pd.DataFrame
    id_map: dict[str, str]
    report: ValidationReport
    dropped_rows: list[int]


def adapt_college_frame(
    frame: pd.DataFrame, curriculum: Curriculum, validation: ValidationResult | None = None
) -> AdaptedData:
    """Validate (if not already done) and convert a college-format table."""
    validation = validation or validate_college_frame(frame, curriculum)
    report = validation.report
    if report.has_fatal_error:
        raise ValueError(
            "The file cannot be processed: "
            + "; ".join(i.message for i in report.of("error") if i.row is None)
        )

    typed = validation.typed.copy()
    for row, column in report.invalid_cells:
        if column in typed.columns:
            typed.at[row - 1, column] = np.nan

    dropped = sorted(report.error_rows)
    keep = ~typed.index.isin([r - 1 for r in dropped])
    source_rows = pd.Series(np.arange(1, len(typed) + 1), index=typed.index)
    typed, source_rows = typed[keep], source_rows[keep]

    identifiers = typed["PRN"].astype(str)
    keys = {prn: f"S{n:06d}" for n, prn in enumerate(pd.unique(identifiers), start=1)}
    student_key = identifiers.map(keys)

    semesters = pd.DataFrame({"student_key": student_key, "source_row": source_rows})
    for column in SEMESTER_COLUMNS:
        semesters[column] = typed[column] if column in typed.columns else np.nan
    semesters["semester"] = semesters["semester"].astype(int)

    subjects = _subject_table(typed, student_key, curriculum)
    return AdaptedData(
        semesters=semesters.sort_values(["student_key", "semester"], ignore_index=True),
        subjects=subjects,
        id_map={key: prn for prn, key in keys.items()},
        report=report,
        dropped_rows=dropped,
    )


def _nan_if_none(value: float | None) -> float:
    return np.nan if value is None else float(value)


def _subject_table(
    typed: pd.DataFrame, student_key: pd.Series, curriculum: Curriculum
) -> pd.DataFrame:
    parts = []
    for code, fields in subject_columns(list(typed.columns)).items():
        spec = curriculum.get(code) or infer_subject_spec(code, fields, typed)
        allowed = set(SCHEME_COMPONENTS[spec.scheme]) | {"attendance_pct", "total"}
        block = pd.DataFrame(
            {
                "student_key": student_key,
                "semester": typed["semester"].astype(int),
                "subject_code": code,
                "scheme": spec.scheme,
                "credits": spec.credits,
                "max_marks": spec.max_marks
                if not np.isnan(spec.max_marks)
                else typed.get(fields.get("max_marks", ""), np.nan),
                "cie_max": _nan_if_none(spec.component_max("cie")),
                "ete_max": _nan_if_none(spec.component_max("ete")),
            }
        )
        for fld in ("attendance_pct", "cie", "ete", "termwork", "total"):
            block[fld] = typed[fields[fld]] if fld in fields and fld in allowed else np.nan
        # Keep only rows that actually contain data for this subject (stacked multi-semester
        # files have empty columns for subjects of other semesters).
        has_data = block[["attendance_pct", "cie", "ete", "termwork", "total"]].notna().any(axis=1)
        parts.append(block[has_data])

    if not parts:
        return pd.DataFrame(columns=SUBJECT_TABLE_COLUMNS)
    table = pd.concat(parts, ignore_index=True)
    table["total_pct"] = table["total"] / table["max_marks"] * 100
    return table[SUBJECT_TABLE_COLUMNS].sort_values(
        ["student_key", "semester", "subject_code"], ignore_index=True
    )
