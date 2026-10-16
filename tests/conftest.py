"""Shared test helpers.

Test data is built in code from the curriculum, so every value is synthetic and the expected
validation outcome of each row is known exactly.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scholartrack.data.curriculum import Curriculum, load_curriculum
from scholartrack.data.loader import read_college_csv


@pytest.fixture(scope="session")
def curriculum() -> Curriculum:
    return load_curriculum()


def valid_row(curriculum: Curriculum, i: int, semester: int = 5) -> dict[str, object]:
    """One internally consistent college-format record with synthetic identifiers."""
    row: dict[str, object] = {
        "PRN": f"SYN{i:05d}",
        "student_name": f"Synthetic Student {i}",
        "class_division": "T.Y.1",
        "academic_year": "2026-27",
        "semester": semester,
        "elective_subject": "Cloud Computing" if semester == 5 else "",
        "age": 20,
        "gender": "Female",
        "prior_sgpa": 7.1,
        "study_hours_per_week": 12.5,
        "commute_time_mins": 20,
        "part_time_job": "No",
    }
    att_sum = pct_sum = credit_sum = 0.0
    for k, spec in enumerate(curriculum.semester_subjects(semester)):
        code, attendance = spec.code, 75.0 + k
        row[f"{code}_attendance_pct"] = attendance
        if spec.scheme == "theory":
            cie, ete = 25, 40
            row[f"{code}_CIE_marks"], row[f"{code}_ETE_marks"] = cie, ete
            total = cie + ete
        elif spec.scheme == "ete_only":
            total = int(spec.max_marks * 0.6)
            row[f"{code}_CIE_marks"], row[f"{code}_ETE_marks"] = 0, total
        else:
            total = int(spec.max_marks * 0.6)
            row[f"{code}_termwork_marks"] = total
        row[f"{code}_total_marks"] = total
        row[f"{code}_max_marks"] = int(spec.max_marks)
        att_sum += spec.credits * attendance
        pct_sum += spec.credits * total / spec.max_marks * 100
        credit_sum += spec.credits
    row["overall_attendance_pct"] = round(att_sum / credit_sum, 2)
    row["credit_weighted_marks_pct"] = round(pct_sum / credit_sum, 2)
    return row


def as_loaded(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Round-trip rows through CSV text, exactly as an uploaded file would arrive."""
    return read_college_csv(pd.DataFrame(rows).to_csv(index=False).encode("utf-8"))
