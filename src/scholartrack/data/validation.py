"""Validation of college-format CSV data.

Validation never edits data. It returns a :class:`ValidationResult` holding

- ``report``: every problem found, as structured :class:`Issue` objects, and
- ``typed``: a copy of the data with numeric columns converted to numbers.

Severities
----------
``error``    The row cannot be used at all (no identifier, no semester, duplicate record),
             or the file lacks a required column.
``invalid``  A value is impossible (e.g. SGPA 12.4 on a 10-point scale, attendance 145 %).
             The adapter treats such a cell as *missing*; it never guesses a "corrected" value.
``warning``  Suspicious but usable (unknown elective name, derived column slightly off,
             an unexpected extra column). Values are kept.

Privacy: issues refer to rows by number only. Values of identifier columns (PRN, name) are
never copied into an issue.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from scholartrack.data.curriculum import SCHEME_COMPONENTS, Curriculum, SubjectSpec
from scholartrack.data.schema import (
    IDENTIFIER_COLUMNS,
    REQUIRED_COLUMNS,
    STUDENT_FIELD_MAP,
    STUDENT_FIELDS,
    parse_subject_column,
    subject_columns,
)

SEVERITIES = ("error", "invalid", "warning")
TOTAL_TOLERANCE = 0.5  # marks
ATTENDANCE_TOLERANCE = 1.0  # percentage points
MARKS_PCT_TOLERANCE = 0.5  # percentage points


@dataclass(frozen=True)
class Issue:
    severity: str
    code: str
    column: str | None
    row: int | None  # 1-based data-row number (spreadsheet line = row + 1); None = whole column
    value: str | None
    message: str


@dataclass
class ValidationReport:
    n_rows: int
    n_columns: int
    issues: list[Issue] = field(default_factory=list)

    def add(
        self,
        severity: str,
        code: str,
        message: str,
        *,
        column: str | None = None,
        row: int | None = None,
        value: object = None,
    ) -> None:
        if severity not in SEVERITIES:
            raise ValueError(severity)
        shown = None if (value is None or column in IDENTIFIER_COLUMNS) else _fmt(value)
        self.issues.append(Issue(severity, code, column, row, shown, message))

    def of(self, severity: str) -> list[Issue]:
        return [i for i in self.issues if i.severity == severity]

    @property
    def has_fatal_error(self) -> bool:
        """A file-level error (row is None) means the file cannot be processed."""
        return any(i.severity == "error" and i.row is None for i in self.issues)

    @property
    def error_rows(self) -> set[int]:
        return {i.row for i in self.issues if i.severity == "error" and i.row is not None}

    @property
    def invalid_cells(self) -> set[tuple[int, str]]:
        return {
            (i.row, i.column)
            for i in self.issues
            if i.severity == "invalid" and i.row is not None and i.column is not None
        }

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [i.__dict__ for i in self.issues],
            columns=["severity", "code", "column", "row", "value", "message"],
        )

    def summary(self) -> pd.DataFrame:
        frame = self.to_frame()
        if frame.empty:
            return pd.DataFrame(columns=["severity", "code", "count"])
        return (
            frame.groupby(["severity", "code"])
            .size()
            .rename("count")
            .reset_index()
            .sort_values(["severity", "count"], ascending=[True, False], ignore_index=True)
        )


@dataclass
class ValidationResult:
    report: ValidationReport
    typed: pd.DataFrame


def _fmt(value: object) -> str:
    if isinstance(value, float | np.floating):
        return f"{value:g}"
    return str(value)


def _rows(mask: pd.Series) -> Iterable[int]:
    """Positional indices where a boolean mask is True."""
    return np.flatnonzero(mask.to_numpy(dtype=bool, na_value=False))


def _to_numeric(frame: pd.DataFrame, column: str, report: ValidationReport) -> pd.Series:
    raw = frame[column]
    numeric = pd.to_numeric(raw, errors="coerce")
    bad = raw.notna() & numeric.isna()
    for i in _rows(bad):
        report.add(
            "invalid",
            "non_numeric",
            f"{column} is not a number",
            column=column,
            row=i + 1,
            value=raw.iloc[i],
        )
    return numeric.astype(float)


def _check_range(
    values: pd.Series,
    column: str,
    lo: float | None,
    hi: float | None,
    report: ValidationReport,
    what: str | None = None,
) -> pd.Series:
    """Flag values outside [lo, hi] as invalid; return the mask of bad values."""
    bad = pd.Series(False, index=values.index)
    if lo is not None:
        bad |= values < lo
    if hi is not None:
        bad |= values > hi
    label = what or column
    for i in _rows(bad):
        report.add(
            "invalid",
            "out_of_range",
            f"{label} = {_fmt(values.iloc[i])} is outside the permitted range "
            f"[{_fmt(lo)}, {_fmt(hi)}]",
            column=column,
            row=i + 1,
            value=values.iloc[i],
        )
    return bad.fillna(False)


def validate_college_frame(frame: pd.DataFrame, curriculum: Curriculum) -> ValidationResult:
    """Validate a college-format table (as returned by ``read_college_csv``)."""
    report = ValidationReport(n_rows=len(frame), n_columns=frame.shape[1])
    frame = frame.reset_index(drop=True)
    typed = frame.copy()

    missing_required = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    for column in missing_required:
        report.add(
            "error",
            "missing_required_column",
            f"Required column '{column}' is absent",
            column=column,
        )
    if missing_required:
        return ValidationResult(report, typed)

    # Columns that are neither known student fields nor subject columns are ignored downstream.
    for column in frame.columns:
        if column not in STUDENT_FIELD_MAP and parse_subject_column(column) is None:
            report.add(
                "warning",
                "unrecognised_column",
                f"Column '{column}' is not part of the expected format and will be ignored",
                column=column,
            )

    _validate_student_fields(frame, typed, report, curriculum)
    _validate_subjects(frame, typed, report, curriculum)
    _validate_derived(typed, report, curriculum)
    return ValidationResult(report, typed)


def _validate_student_fields(
    frame: pd.DataFrame, typed: pd.DataFrame, report: ValidationReport, curriculum: Curriculum
) -> None:
    # Identifier and semester must be present for a row to be usable.
    for i in _rows(frame["PRN"].isna()):
        report.add(
            "error",
            "missing_required_value",
            "Student identifier (PRN) is missing",
            column="PRN",
            row=i + 1,
        )

    for spec in STUDENT_FIELDS:
        if spec.name not in frame.columns or spec.kind == "text" and spec.pattern is None:
            continue
        column = spec.name
        if spec.kind in ("numeric", "integer"):
            values = _to_numeric(frame, column, report)
            if spec.kind == "integer":
                fractional = values.notna() & (values % 1 != 0)
                for i in _rows(fractional):
                    report.add(
                        "invalid",
                        "not_integer",
                        f"{column} must be a whole number",
                        column=column,
                        row=i + 1,
                        value=values.iloc[i],
                    )
            _check_range(values, column, spec.minimum, spec.maximum, report)
            typed[column] = values
        elif spec.allowed:
            values = frame[column]
            bad = values.notna() & ~values.isin(spec.allowed)
            for i in _rows(bad):
                report.add(
                    "invalid",
                    "unknown_category",
                    f"{column} = '{values.iloc[i]}' is not one of {list(spec.allowed)}",
                    column=column,
                    row=i + 1,
                    value=values.iloc[i],
                )
        elif spec.pattern is not None:
            values = frame[column]
            bad = values.notna() & ~values.str.fullmatch(spec.pattern).fillna(False)
            for i in _rows(bad):
                report.add(
                    "warning",
                    "unexpected_format",
                    f"{column} = '{values.iloc[i]}' does not match the expected format",
                    column=column,
                    row=i + 1,
                    value=values.iloc[i],
                )

    semester = typed["semester"]
    bad_semester = semester.isna() | (semester % 1 != 0) | (semester < 1) | (semester > 8)
    for i in _rows(bad_semester):
        report.add(
            "error",
            "missing_required_value",
            "Semester is missing or invalid, so the row cannot be placed in time",
            column="semester",
            row=i + 1,
        )

    key = frame["PRN"].astype("string") + "|" + semester.astype("string")
    duplicated = key.notna() & key.duplicated(keep=False)
    first_seen: dict[str, int] = {}
    for i in _rows(duplicated):
        k = key.iloc[i]
        if k in first_seen:
            report.add(
                "error",
                "duplicate_record",
                f"Same student identifier and semester as row {first_seen[k]}",
                column="PRN",
                row=i + 1,
            )
        else:
            first_seen[k] = i + 1
            report.add(
                "error",
                "duplicate_record",
                "Student identifier and semester appear more than once",
                column="PRN",
                row=i + 1,
            )

    if "elective_subject" in frame.columns:
        electives = frame["elective_subject"]
        for i in _rows(electives.notna()):
            sem = semester.iloc[i]
            options = curriculum.elective_options.get(int(sem), ()) if pd.notna(sem) else ()
            if options and electives.iloc[i] not in options:
                report.add(
                    "warning",
                    "unknown_elective",
                    f"Elective '{electives.iloc[i]}' is not in the curriculum options "
                    f"for semester {int(sem)}",
                    column="elective_subject",
                    row=i + 1,
                    value=electives.iloc[i],
                )


def infer_subject_spec(code: str, fields: dict[str, str], typed: pd.DataFrame) -> SubjectSpec:
    """Best-effort scheme for a subject that is not in the curriculum (credits unknown)."""

    def has_data(fld: str) -> bool:
        return fld in fields and typed[fields[fld]].notna().any()

    if has_data("termwork") and not has_data("ete"):
        scheme = "termwork"
    elif has_data("ete") and (not has_data("cie") or (typed[fields["cie"]].fillna(0) == 0).all()):
        scheme = "ete_only"
    else:
        scheme = "theory"
    max_marks = typed[fields["max_marks"]].max() if "max_marks" in fields else np.nan
    return SubjectSpec(
        code=code,
        semester=0,
        credits=np.nan,
        scheme=scheme,
        max_marks=float(max_marks),
        cie_max=None,
        ete_max=None,
    )


def _validate_subjects(
    frame: pd.DataFrame, typed: pd.DataFrame, report: ValidationReport, curriculum: Curriculum
) -> None:
    for code, fields in subject_columns(list(frame.columns)).items():
        for column in fields.values():
            typed[column] = _to_numeric(frame, column, report)

        spec = curriculum.get(code)
        if spec is None:
            spec = infer_subject_spec(code, fields, typed)
            report.add(
                "warning",
                "unknown_subject",
                f"Subject {code} is not in the curriculum; scheme inferred as "
                f"'{spec.scheme}', credits unknown",
                column=code,
            )

        expected = set(SCHEME_COMPONENTS[spec.scheme]) | {"attendance_pct", "total", "max_marks"}
        for fld, column in fields.items():
            if fld in expected:
                continue
            if fld == "cie" and spec.scheme == "ete_only":
                nonzero = typed[column].notna() & (typed[column] != 0)
                for i in _rows(nonzero):
                    report.add(
                        "warning",
                        "unexpected_component_value",
                        f"{code} is assessed by ETE only, but CIE = {_fmt(typed[column].iloc[i])}",
                        column=column,
                        row=i + 1,
                        value=typed[column].iloc[i],
                    )
                continue
            n_values = int(typed[column].notna().sum())
            report.add(
                "warning",
                "unexpected_column",
                f"{column} does not belong to a '{spec.scheme}' subject; it has "
                f"{n_values} non-empty value(s) out of {len(frame)} and will be ignored",
                column=column,
            )
        for fld in sorted(expected - set(fields) - {"max_marks"}):
            report.add(
                "warning", "missing_component_column", f"{code} has no '{fld}' column", column=code
            )

        max_marks = typed[fields["max_marks"]] if "max_marks" in fields else None
        if max_marks is not None and not np.isnan(spec.max_marks):
            bad_max = max_marks.notna() & (max_marks != spec.max_marks)
            for i in _rows(bad_max):
                report.add(
                    "invalid",
                    "max_marks_mismatch",
                    f"{fields['max_marks']} = {_fmt(max_marks.iloc[i])}, curriculum says "
                    f"{_fmt(spec.max_marks)}",
                    column=fields["max_marks"],
                    row=i + 1,
                    value=max_marks.iloc[i],
                )

        if "attendance_pct" in fields:
            _check_range(typed[fields["attendance_pct"]], fields["attendance_pct"], 0, 100, report)

        component_bad = pd.Series(False, index=typed.index)
        for comp in SCHEME_COMPONENTS[spec.scheme]:
            if comp not in fields:
                continue
            cmax = spec.component_max(comp)
            if cmax is None or np.isnan(cmax):
                cmax = spec.max_marks
            component_bad |= _check_range(typed[fields[comp]], fields[comp], 0, cmax, report)

        if "total" in fields:
            total_bad = _check_range(
                typed[fields["total"]], fields["total"], 0, spec.max_marks, report
            )
            comps = [fields[c] for c in SCHEME_COMPONENTS[spec.scheme] if c in fields]
            if comps and len(comps) == len(SCHEME_COMPONENTS[spec.scheme]):
                parts = typed[comps]
                summed = parts.sum(axis=1, min_count=len(comps))
                mismatch = (
                    ((summed - typed[fields["total"]]).abs() > TOTAL_TOLERANCE)
                    & ~component_bad
                    & ~total_bad
                )
                for i in _rows(mismatch):
                    report.add(
                        "warning",
                        "total_mismatch",
                        f"{fields['total']} = {_fmt(typed[fields['total']].iloc[i])} but "
                        f"components add up to {_fmt(summed.iloc[i])}",
                        column=fields["total"],
                        row=i + 1,
                        value=typed[fields["total"]].iloc[i],
                    )


def credit_weighted(values: pd.DataFrame, credits: dict[str, float]) -> pd.Series:
    """Credit-weighted row mean over subject columns; NaN if any needed value is missing."""
    weights = np.array([credits[c] for c in values.columns], dtype=float)
    complete = values.notna().all(axis=1)
    result = (values.to_numpy(dtype=float) * weights).sum(axis=1) / weights.sum()
    return pd.Series(np.where(complete, result, np.nan), index=values.index)


def _validate_derived(
    typed: pd.DataFrame, report: ValidationReport, curriculum: Curriculum
) -> None:
    """Check reported summary columns against values recomputed from subject columns."""
    invalid = report.invalid_cells
    grouped = subject_columns(list(typed.columns))
    for sem in typed["semester"].dropna().unique():
        rows = typed.index[typed["semester"] == sem]
        specs = [s for s in curriculum.semester_subjects(int(sem)) if s.code in grouped]
        if not specs:
            continue
        credits = {s.code: s.credits for s in specs}
        checks = [
            ("overall_attendance_pct", "attendance_pct", ATTENDANCE_TOLERANCE, None),
            ("credit_weighted_marks_pct", "total", MARKS_PCT_TOLERANCE, "pct"),
        ]
        for target, fld, tol, scale in checks:
            if target not in typed.columns or any(fld not in grouped[s.code] for s in specs):
                continue
            cols = {s.code: grouped[s.code][fld] for s in specs}
            values = pd.DataFrame({code: typed.loc[rows, col] for code, col in cols.items()})
            if scale == "pct":
                values = pd.DataFrame({s.code: values[s.code] / s.max_marks * 100 for s in specs})
            # Ignore rows where any input cell (or the target) is already invalid.
            usable = pd.Series(True, index=rows)
            for r in rows:
                if (r + 1, target) in invalid or any((r + 1, c) in invalid for c in cols.values()):
                    usable[r] = False
            recomputed = credit_weighted(values, credits)
            diff = (typed.loc[rows, target] - recomputed).abs()
            bad = usable & (diff > tol)
            for r in rows[bad.to_numpy(dtype=bool, na_value=False)]:
                report.add(
                    "warning",
                    "derived_mismatch",
                    f"{target} = {_fmt(typed.at[r, target])} but recomputing it from the "
                    f"subject columns gives {recomputed[r]:.2f}",
                    column=target,
                    row=int(r) + 1,
                    value=typed.at[r, target],
                )
