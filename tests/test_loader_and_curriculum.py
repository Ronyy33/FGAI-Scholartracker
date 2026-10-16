"""CSV loading edge cases and curriculum / grading rules."""

from __future__ import annotations

import numpy as np
import pytest

from scholartrack.data.loader import CsvFormatError, read_college_csv


def test_byte_order_mark_is_removed():
    frame = read_college_csv("﻿PRN,semester\nSYN1,5\n".encode())
    assert list(frame.columns) == ["PRN", "semester"]


def test_blank_cells_and_whitespace():
    frame = read_college_csv(b"PRN,semester,gender\n SYN1 ,5,\n")
    assert frame.at[0, "PRN"] == "SYN1"
    assert np.isnan(frame.at[0, "gender"])


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"", "empty"),
        (b"PRN,semester\n", "no data rows"),
        (b"PRN,PRN\n1,2\n", "Duplicate column"),
        (b"PRN,,semester\n1,2,3\n", "empty column name"),
        (b"\xff\xfe\x00P", "not UTF-8"),
    ],
)
def test_malformed_files_are_rejected(payload, message):
    with pytest.raises(CsvFormatError, match=message):
        read_college_csv(payload)


def test_size_limits():
    payload = b"PRN,semester\n" + b"SYN1,5\n" * 10
    with pytest.raises(CsvFormatError, match="rows"):
        read_college_csv(payload, max_rows=5)
    with pytest.raises(CsvFormatError, match="bytes"):
        read_college_csv(payload, max_bytes=20)


def test_semester_five_matches_reference_structure(curriculum):
    subjects = {s.code: s for s in curriculum.semester_subjects(5)}
    assert sum(s.credits for s in subjects.values()) == 23
    assert subjects["CE24PCC301"].scheme == "theory"
    assert subjects["CE24PCC301"].component_max("cie") == 40
    assert subjects["CE24OEL310"].scheme == "ete_only"
    assert subjects["CE24OEL310"].component_max("ete") == 50
    assert subjects["CE24OEL310"].component_max("cie") is None
    assert subjects["CE24PEC304"].scheme == "termwork"


def test_every_semester_is_defined(curriculum):
    assert curriculum.semesters == list(range(1, 9))
    for semester in curriculum.semesters:
        assert 18 <= sum(s.credits for s in curriculum.semester_subjects(semester)) <= 24


@pytest.mark.parametrize(
    ("pct", "passed", "points"),
    [
        (95, True, 10),
        (90, True, 10),
        (89.9, True, 9),
        (60, True, 7),
        (40, True, 4),
        (39.9, True, 0),
        (75, False, 0),
    ],
)
def test_grade_points(curriculum, pct, passed, points):
    result = curriculum.grading.points_for(np.array([pct]), np.array([passed]))
    assert result[0] == points


def test_grade_points_keep_missing_values(curriculum):
    result = curriculum.grading.points_for(np.array([np.nan]), np.array([True]))
    assert np.isnan(result[0])
