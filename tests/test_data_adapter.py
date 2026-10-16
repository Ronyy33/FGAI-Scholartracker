"""The adapter must drop identifiers, blank invalid cells and keep everything else intact."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scholartrack.data.adapter import adapt_college_frame
from tests.conftest import as_loaded, valid_row


def test_identifiers_removed_and_mapped(curriculum):
    adapted = adapt_college_frame(
        as_loaded([valid_row(curriculum, i) for i in range(3)]), curriculum
    )
    for table in (adapted.semesters, adapted.subjects):
        assert "PRN" not in table.columns and "student_name" not in table.columns
        assert not table.astype(str).apply(lambda c: c.str.contains("SYN0")).any().any()
    assert adapted.id_map == {"S000001": "SYN00000", "S000002": "SYN00001", "S000003": "SYN00002"}


def test_subject_table_shape_and_derived_values(curriculum):
    adapted = adapt_college_frame(
        as_loaded([valid_row(curriculum, i) for i in range(4)]), curriculum
    )
    n_subjects = len(curriculum.semester_subjects(5))
    assert len(adapted.subjects) == 4 * n_subjects
    row = adapted.subjects.query("student_key == 'S000001' and subject_code == 'CE24PCC301'")
    assert row["credits"].item() == 3
    assert row["total_pct"].item() == pytest.approx(65.0)
    # ETE-only subjects: the CIE column is "not applicable", so it is not carried over.
    oel = adapted.subjects.query("subject_code == 'CE24OEL310'")
    assert oel["cie"].isna().all() and oel["ete"].notna().all()


def test_invalid_cells_become_missing_other_values_kept(curriculum):
    rows = [valid_row(curriculum, i) for i in range(2)]
    rows[0]["prior_sgpa"] = 12.4
    rows[0]["CE24PCC302_CIE_marks"] = 55
    adapted = adapt_college_frame(as_loaded(rows), curriculum)
    sem = adapted.semesters.set_index("student_key")
    assert np.isnan(sem.at["S000001", "prior_sgpa"])
    assert sem.at["S000002", "prior_sgpa"] == pytest.approx(7.1)
    subj = adapted.subjects.set_index(["student_key", "subject_code"])
    assert np.isnan(subj.at[("S000001", "CE24PCC302"), "cie"])
    # The total is kept: it was valid, and no value is "repaired" by guessing.
    assert subj.at[("S000001", "CE24PCC302"), "total"] == 65


def test_error_rows_are_dropped_and_reported(curriculum):
    rows = [valid_row(curriculum, i) for i in range(3)]
    rows[1]["semester"] = None
    adapted = adapt_college_frame(as_loaded(rows), curriculum)
    assert adapted.dropped_rows == [2]
    assert list(adapted.semesters["source_row"]) == [1, 3]


def test_stacked_multi_semester_file(curriculum):
    rows = [valid_row(curriculum, 0, semester=4), valid_row(curriculum, 0, semester=5)]
    frame = pd.DataFrame(rows)  # union of both semesters' columns; others left empty
    adapted = adapt_college_frame(as_loaded(frame.to_dict("records")), curriculum)
    assert list(adapted.semesters["semester"]) == [4, 5]
    per_semester = adapted.subjects.groupby("semester")["subject_code"].nunique().to_dict()
    assert per_semester == {
        4: len(curriculum.semester_subjects(4)),
        5: len(curriculum.semester_subjects(5)),
    }


def test_fatal_file_raises(curriculum):
    rows = [valid_row(curriculum, 0)]
    del rows[0]["PRN"]
    with pytest.raises(ValueError, match="cannot be processed"):
        adapt_college_frame(as_loaded(rows), curriculum)
