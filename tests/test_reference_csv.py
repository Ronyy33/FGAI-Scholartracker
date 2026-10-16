"""Checks against the real reference file.

The file is never committed (public repository), so these tests are skipped when it is absent,
for example in CI. Locally they pin down exactly which problems the file contains.
"""

from __future__ import annotations

import hashlib

import pytest

from scholartrack.data.adapter import adapt_college_frame
from scholartrack.data.loader import read_college_csv
from scholartrack.data.validation import validate_college_frame
from scholartrack.paths import REFERENCE_CSV

REFERENCE_SHA256 = "2746d8466cbe93159307b2f178f6894f442226b218162ea5a876d38b5201559a"

pytestmark = pytest.mark.skipif(not REFERENCE_CSV.exists(), reason="reference CSV not present")


def test_reference_file_is_unmodified():
    assert hashlib.sha256(REFERENCE_CSV.read_bytes()).hexdigest() == REFERENCE_SHA256


def test_reference_issues(curriculum):
    report = validate_college_frame(read_college_csv(REFERENCE_CSV), curriculum).report
    found = {(i.code, i.column, i.row) for i in report.issues}
    expected = {
        ("out_of_range", "prior_sgpa", 17),
        ("out_of_range", "study_hours_per_week", 122),
        ("out_of_range", "commute_time_mins", 33),
        ("out_of_range", "overall_attendance_pct", 180),
        ("out_of_range", "CE24PCC301_attendance_pct", 104),
        ("out_of_range", "CE24PCC301_CIE_marks", 172),
        ("out_of_range", "CE24PCC302_CIE_marks", 9),
        ("out_of_range", "CE24PCC303_ETE_marks", 134),
        ("out_of_range", "CE24PCC308_termwork_marks", 61),
        ("out_of_range", "CE24MDM309_termwork_marks", 151),
        ("unknown_category", "gender", 49),
        ("unknown_category", "part_time_job", 90),
        ("unknown_elective", "elective_subject", 166),
        ("unexpected_column", "CE24OEL310_termwork_marks", None),
        ("unexpected_column", "CE24AEC311_termwork_marks", None),
    }
    assert found == expected
    assert not report.error_rows


def test_reference_adapts_cleanly(curriculum):
    adapted = adapt_college_frame(read_college_csv(REFERENCE_CSV), curriculum)
    assert len(adapted.semesters) == 200
    assert len(adapted.subjects) == 200 * 11
    assert adapted.semesters["prior_sgpa"].max() <= 10


def test_quality_report_contains_no_identifiers():
    import re
    import sys

    sys.path.insert(0, str(REFERENCE_CSV.parents[2] / "scripts"))
    from inspect_reference_data import build_report

    text = build_report(REFERENCE_CSV)
    raw = read_college_csv(REFERENCE_CSV)
    assert not any(prn in text for prn in raw["PRN"])
    surnames = {name.split()[-1] for name in raw["student_name"].dropna()}
    assert not any(re.search(rf"\b{re.escape(s)}\b", text) for s in surnames)
    assert build_report(REFERENCE_CSV) == text  # deterministic
