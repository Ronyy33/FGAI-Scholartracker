"""Validation must catch every class of problem found in the reference CSV, and nothing else."""

from __future__ import annotations

import pytest

from scholartrack.data.validation import validate_college_frame
from tests.conftest import as_loaded, valid_row


def _issues(curriculum, rows):
    return validate_college_frame(as_loaded(rows), curriculum).report


def test_clean_data_produces_no_issues(curriculum):
    report = _issues(curriculum, [valid_row(curriculum, i) for i in range(3)])
    assert report.issues == []


@pytest.mark.parametrize(
    ("column", "value", "code", "severity"),
    [
        ("prior_sgpa", 12.4, "out_of_range", "invalid"),
        ("study_hours_per_week", -4, "out_of_range", "invalid"),
        ("commute_time_mins", -15, "out_of_range", "invalid"),
        ("overall_attendance_pct", 132, "out_of_range", "invalid"),
        ("CE24PCC301_attendance_pct", 145, "out_of_range", "invalid"),
        ("CE24PCC301_CIE_marks", -5, "out_of_range", "invalid"),
        ("CE24PCC302_CIE_marks", 55, "out_of_range", "invalid"),
        ("CE24PCC303_ETE_marks", 78, "out_of_range", "invalid"),
        ("CE24MDM309_termwork_marks", 35, "out_of_range", "invalid"),
        ("CE24PCC301_total_marks", 101, "out_of_range", "invalid"),
        ("CE24PCC301_max_marks", 50, "max_marks_mismatch", "invalid"),
        ("gender", "Unknown??", "unknown_category", "invalid"),
        ("part_time_job", "Sometimes", "unknown_category", "invalid"),
        ("study_hours_per_week", "12a", "non_numeric", "invalid"),
        ("age", 20.5, "not_integer", "invalid"),
        ("elective_subject", "Cybersecurity++", "unknown_elective", "warning"),
        ("academic_year", "2026/27", "unexpected_format", "warning"),
        ("semester", 9, "missing_required_value", "error"),
        ("PRN", "", "missing_required_value", "error"),
    ],
)
def test_injected_problem_is_detected(curriculum, column, value, code, severity):
    rows = [valid_row(curriculum, i) for i in range(3)]
    rows[1][column] = value
    report = _issues(curriculum, rows)
    matches = [i for i in report.issues if i.code == code and i.column == column and i.row == 2]
    assert matches, report.to_frame().to_string()
    assert matches[0].severity == severity


def test_component_error_does_not_also_report_total_mismatch(curriculum):
    rows = [valid_row(curriculum, 0)]
    rows[0]["CE24PCC301_CIE_marks"] = -5  # total left as it was, like the reference file
    codes = {i.code for i in _issues(curriculum, rows).issues}
    assert codes == {"out_of_range"}


def test_total_mismatch_with_valid_components_is_a_warning(curriculum):
    rows = [valid_row(curriculum, 0)]
    rows[0]["CE24PCC301_total_marks"] = 70  # components add up to 65
    report = _issues(curriculum, rows)
    # +5 marks on a 3-credit subject also moves credit_weighted_marks_pct by 5*3/23 = 0.65
    # points, beyond its 0.5 tolerance, so the derived column is (correctly) flagged too.
    assert sorted((i.code, i.column, i.severity) for i in report.issues) == [
        ("derived_mismatch", "credit_weighted_marks_pct", "warning"),
        ("total_mismatch", "CE24PCC301_total_marks", "warning"),
    ]


def test_derived_attendance_mismatch_is_detected(curriculum):
    rows = [valid_row(curriculum, 0)]
    rows[0]["overall_attendance_pct"] = float(rows[0]["overall_attendance_pct"]) + 5
    report = _issues(curriculum, rows)
    assert [(i.code, i.column) for i in report.issues] == [
        ("derived_mismatch", "overall_attendance_pct")
    ]


def test_derived_check_skips_rows_with_invalid_inputs(curriculum):
    rows = [valid_row(curriculum, 0)]
    rows[0]["CE24PCC301_attendance_pct"] = 145  # makes the recomputed mean meaningless
    codes = [i.code for i in _issues(curriculum, rows).issues]
    assert codes == ["out_of_range"]


def test_structural_column_problems(curriculum):
    rows = [valid_row(curriculum, i) for i in range(2)]
    for row in rows:
        row["CE24OEL310_termwork_marks"] = None  # stray, empty column
        row["favourite_colour"] = "blue"
        row["CE24XYZ999_attendance_pct"] = 80
        row["CE24XYZ999_termwork_marks"] = 10
        row["CE24XYZ999_total_marks"] = 10
        row["CE24XYZ999_max_marks"] = 25
    rows[0]["CE24OEL310_CIE_marks"] = 5  # ETE-only subject must have CIE 0 / empty
    found = {(i.code, i.column) for i in _issues(curriculum, rows).issues}
    assert ("unexpected_column", "CE24OEL310_termwork_marks") in found
    assert ("unrecognised_column", "favourite_colour") in found
    assert ("unknown_subject", "CE24XYZ999") in found
    assert ("unexpected_component_value", "CE24OEL310_CIE_marks") in found


def test_missing_required_column_is_fatal(curriculum):
    rows = [valid_row(curriculum, 0)]
    del rows[0]["semester"]
    report = _issues(curriculum, rows)
    assert report.has_fatal_error


def test_duplicates_are_errors_and_identifiers_never_leak(curriculum):
    rows = [valid_row(curriculum, 0), valid_row(curriculum, 1), valid_row(curriculum, 0)]
    rows[1]["gender"] = "Unknown??"
    report = _issues(curriculum, rows)
    duplicates = [i for i in report.issues if i.code == "duplicate_record"]
    assert {i.row for i in duplicates} == {1, 3}
    assert all(i.severity == "error" for i in duplicates)
    text = report.to_frame().to_string()
    assert "SYN00000" not in text and "Synthetic Student" not in text
