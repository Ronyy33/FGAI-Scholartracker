"""Schema of the college-format CSV (the format of the reference file).

The format is *wide*: one row per student per semester, with student-level columns
(``PRN``, ``prior_sgpa``, ...) followed by five or so columns per subject, named
``<SUBJECTCODE>_<field>`` (for example ``CE24PCC301_CIE_marks``).

Each student-level column has a *role*, which later phases use to decide what may become a
model input:

- ``identifier``  : direct identifier; never a model input, never logged or reported.
- ``context``     : describes the record (semester, year, division); not a model input.
- ``sensitive``   : demographic; excluded from model inputs by default (fairness review).
- ``candidate``   : may become a model input if available at prediction time.
- ``outcome``     : measured at the END of the semester (contains end-term results).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

ROLES = ("identifier", "context", "sensitive", "candidate", "outcome")


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: str  # "numeric" | "integer" | "categorical" | "text"
    role: str
    required: bool = False
    minimum: float | None = None
    maximum: float | None = None
    allowed: tuple[str, ...] = field(default_factory=tuple)
    pattern: str | None = None
    unit: str = ""
    description: str = ""


STUDENT_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec(
        "PRN",
        "text",
        "identifier",
        required=True,
        description="Permanent registration number (direct identifier)",
    ),
    FieldSpec("student_name", "text", "identifier", description="Student name (direct identifier)"),
    FieldSpec(
        "class_division",
        "text",
        "context",
        pattern=r"^[A-Z]{1,2}\.Y\.\d+$",
        description="Class and division, e.g. T.Y.1",
    ),
    FieldSpec(
        "academic_year",
        "text",
        "context",
        pattern=r"^\d{4}-\d{2}$",
        description="Academic year, e.g. 2026-27",
    ),
    FieldSpec(
        "semester",
        "integer",
        "context",
        required=True,
        minimum=1,
        maximum=8,
        description="Semester number of this record",
    ),
    FieldSpec(
        "elective_subject",
        "categorical",
        "context",
        description="Chosen programme elective (values checked against the curriculum)",
    ),
    FieldSpec(
        "age",
        "integer",
        "sensitive",
        minimum=15,
        maximum=40,
        unit="years",
        description="Age in years",
    ),
    FieldSpec(
        "gender",
        "categorical",
        "sensitive",
        allowed=("Female", "Male", "Other", "Prefer not to say"),
        description="Self-reported gender",
    ),
    FieldSpec(
        "prior_sgpa",
        "numeric",
        "candidate",
        minimum=0,
        maximum=10,
        unit="SGPA (0-10)",
        description="SGPA of the previous semester",
    ),
    FieldSpec(
        "study_hours_per_week",
        "numeric",
        "candidate",
        minimum=0,
        maximum=100,
        unit="hours/week",
        description="Self-reported weekly study hours",
    ),
    FieldSpec(
        "commute_time_mins",
        "numeric",
        "candidate",
        minimum=0,
        maximum=300,
        unit="minutes",
        description="One-way commute time",
    ),
    FieldSpec(
        "part_time_job",
        "categorical",
        "candidate",
        allowed=("Yes", "No"),
        description="Whether the student has a part-time job",
    ),
    FieldSpec(
        "overall_attendance_pct",
        "numeric",
        "candidate",
        minimum=0,
        maximum=100,
        unit="%",
        description="Credit-weighted mean of subject attendance",
    ),
    FieldSpec(
        "credit_weighted_marks_pct",
        "numeric",
        "outcome",
        minimum=0,
        maximum=100,
        unit="%",
        description="Credit-weighted mean of subject total % (includes ETE)",
    ),
)

STUDENT_FIELD_MAP: dict[str, FieldSpec] = {f.name: f for f in STUDENT_FIELDS}
IDENTIFIER_COLUMNS: tuple[str, ...] = tuple(
    f.name for f in STUDENT_FIELDS if f.role == "identifier"
)
REQUIRED_COLUMNS: tuple[str, ...] = tuple(f.name for f in STUDENT_FIELDS if f.required)

# Subject columns: <CODE>_<field>. Codes look like CE24PCC301 (letters/digits only).
SUBJECT_FIELDS: dict[str, str] = {
    # CSV suffix -> canonical component name
    "attendance_pct": "attendance_pct",
    "CIE_marks": "cie",
    "ETE_marks": "ete",
    "termwork_marks": "termwork",
    "total_marks": "total",
    "max_marks": "max_marks",
}
SUBJECT_COLUMN_RE = re.compile(
    r"^(?P<code>[A-Z]{2}\d{2}[A-Z]{2,4}\d{3})_(?P<field>"
    + "|".join(re.escape(k) for k in SUBJECT_FIELDS)
    + r")$"
)

# Where each subject field sits in time: CIE, attendance and termwork accrue during the
# semester; ETE and total exist only after end-term exams (they are outcomes of the semester).
SUBJECT_FIELD_ROLES: dict[str, str] = {
    "attendance_pct": "candidate",
    "cie": "candidate",
    "termwork": "outcome",
    "ete": "outcome",
    "total": "outcome",
    "max_marks": "context",
}


def parse_subject_column(column: str) -> tuple[str, str] | None:
    """Return (subject_code, canonical_field) for a subject column, else None."""
    match = SUBJECT_COLUMN_RE.match(column)
    if match is None:
        return None
    return match["code"], SUBJECT_FIELDS[match["field"]]


def subject_columns(columns: list[str]) -> dict[str, dict[str, str]]:
    """Group subject columns by subject code: {code: {canonical_field: column_name}}."""
    grouped: dict[str, dict[str, str]] = {}
    for column in columns:
        parsed = parse_subject_column(column)
        if parsed is not None:
            code, fld = parsed
            grouped.setdefault(code, {})[fld] = column
    return grouped
