"""Synthetic data: reproducible, schema-valid, bounded, consistent, and label-free."""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd
import pytest

from scholartrack.data.adapter import adapt_college_frame
from scholartrack.data.loader import read_college_csv
from scholartrack.data.validation import validate_college_frame
from scholartrack.paths import REFERENCE_CSV
from scholartrack.synthetic.generator import generate, load_synthetic_config, to_college_format
from scholartrack.targets import (
    load_target_config,
    next_semester_labels,
    semester_outcomes,
    subject_outcomes,
)


@pytest.fixture(scope="module")
def target():
    return load_target_config()


@pytest.fixture(scope="module")
def dataset(curriculum, target):
    return generate(load_synthetic_config(), curriculum, target)


@pytest.fixture(scope="module")
def wide(dataset, curriculum):
    return to_college_format(dataset, curriculum)


def _csv_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


def test_same_seed_same_data_different_seed_different_data(dataset, curriculum, target):
    again = generate(load_synthetic_config(), curriculum, target)
    for name in ("semesters", "subjects", "latent", "students"):
        assert _csv_hash(getattr(dataset, name)) == _csv_hash(getattr(again, name)), name
    cfg = load_synthetic_config()
    cfg["seed"] += 1
    other = generate(cfg, curriculum, target)
    assert _csv_hash(other.subjects) != _csv_hash(dataset.subjects)


def test_size_and_cohort_structure(dataset):
    assert 5_000 <= len(dataset.semesters) <= 10_000
    last = dataset.semesters.merge(dataset.students[["student_key", "cohort"]], on="student_key")
    max_sem = last.groupby("cohort")["semester"].max().to_dict()
    assert max_sem == {2019: 8, 2020: 8, 2021: 8, 2022: 8, 2023: 7, 2024: 5}
    latest = dataset.semesters.query("semester == 5 and academic_year == '2026-27'")
    assert len(latest) > 150  # the cohort that mirrors the reference file


def test_exports_pass_validation_without_any_issue(wide, curriculum):
    for (year, semester), frame in wide.items():
        loaded = read_college_csv(frame.to_csv(index=False).encode())
        report = validate_college_frame(loaded, curriculum).report
        assert report.issues == [], (year, semester, report.to_frame().head())


def test_marks_are_written_as_whole_numbers(wide):
    text = wide[("2026-27", 5)].to_csv(index=False)
    header, *lines = text.splitlines()
    columns = header.split(",")
    mark_columns = [i for i, c in enumerate(columns) if c.endswith("_marks")]
    for line in lines:
        cells = line.split(",")
        assert not [cells[i] for i in mark_columns if cells[i].endswith(".0")]


def test_semester_five_export_has_reference_layout(wide):
    columns = list(wide[("2026-27", 5)].columns)
    if REFERENCE_CSV.exists():
        reference = list(read_college_csv(REFERENCE_CSV).columns)
        stray = ["CE24AEC311_termwork_marks", "CE24OEL310_termwork_marks"]
        assert columns == [c for c in reference if c not in stray]
    assert columns[:3] == ["PRN", "student_name", "class_division"]
    assert columns[-2:] == ["overall_attendance_pct", "credit_weighted_marks_pct"]


def test_round_trip_through_adapter_reproduces_internal_tables(dataset, wide, curriculum):
    stacked = pd.concat(wide.values(), ignore_index=True)
    adapted = adapt_college_frame(
        read_college_csv(stacked.to_csv(index=False).encode()), curriculum
    )
    prn_to_key = dict(zip(dataset.students["prn"], dataset.students["student_key"], strict=True))
    to_generator_key = {k: prn_to_key[prn] for k, prn in adapted.id_map.items()}

    sem = adapted.semesters.drop(columns="source_row")
    sem["student_key"] = sem["student_key"].map(to_generator_key)
    sem = sem.sort_values(["student_key", "semester"], ignore_index=True)
    expected = dataset.semesters
    assert list(sem.columns) == list(expected.columns)
    for column in expected.columns:
        a, b = sem[column], expected[column]
        if pd.api.types.is_numeric_dtype(b):
            np.testing.assert_allclose(
                a.astype(float), b.astype(float), equal_nan=True, err_msg=column
            )
        else:
            assert a.fillna("<NA>").astype(str).tolist() == b.fillna("<NA>").astype(str).tolist()

    subj = adapted.subjects.copy()
    subj["student_key"] = subj["student_key"].map(to_generator_key)
    subj = subj.sort_values(["student_key", "semester", "subject_code"], ignore_index=True)
    exp = dataset.subjects
    assert len(subj) == len(exp)
    for column in ("credits", "max_marks", "attendance_pct", "cie", "ete", "termwork", "total"):
        np.testing.assert_allclose(
            subj[column].astype(float), exp[column].astype(float), equal_nan=True, err_msg=column
        )


def test_values_are_bounded(dataset):
    sem, subj = dataset.semesters, dataset.subjects
    assert sem["prior_sgpa"].dropna().between(0, 10).all()
    assert sem["study_hours_per_week"].dropna().between(0, 40).all()
    assert sem["overall_attendance_pct"].dropna().between(0, 100).all()
    assert subj["attendance_pct"].dropna().between(20, 100).all()
    assert (subj["total"].dropna() <= subj.loc[subj["total"].notna(), "max_marks"]).all()
    assert (subj[["cie", "ete", "termwork", "total"]].fillna(0) >= 0).all().all()
    theory = subj[subj["scheme"] == "theory"].dropna(subset=["cie", "ete", "total"])
    assert (theory["cie"] + theory["ete"] == theory["total"]).all()


def test_prior_sgpa_is_previous_semester_sgpa(dataset):
    lat = dataset.latent.set_index(["student_key", "semester"])["sgpa_complete_data"]
    sem = dataset.semesters.dropna(subset=["prior_sgpa"])
    previous = lat.reindex(list(zip(sem["student_key"], sem["semester"] - 1, strict=True)))
    np.testing.assert_allclose(sem["prior_sgpa"].to_numpy(), previous.round(2).to_numpy())
    assert dataset.semesters.query("semester == 1")["prior_sgpa"].isna().all()


def test_no_label_or_latent_columns_in_model_facing_tables(dataset, wide):
    forbidden = re.compile(r"label|risk|difficulty|ability|engagement|trajectory|trend|shock")
    for frame in [dataset.semesters, dataset.subjects, *wide.values()]:
        assert not [c for c in frame.columns if forbidden.search(c)]


def test_identifiers_are_synthetic(dataset):
    assert dataset.students["prn"].str.fullmatch(r"SYN\d{2}CE\d{4}").all()
    assert dataset.students["student_name"].str.startswith("Synthetic Student ").all()
    if REFERENCE_CSV.exists():
        reference = read_college_csv(REFERENCE_CSV)
        assert not set(reference["PRN"]) & set(dataset.students["prn"])
        assert not set(reference["student_name"]) & set(dataset.students["student_name"])


def test_edge_cases_are_present(dataset, curriculum, target):
    subj, lat = dataset.subjects, dataset.latent
    assert (subj["attendance_pct"] == 100).any()
    assert ((subj["scheme"] == "theory") & (subj["ete"] == 0)).any()  # absent from ETE
    assert lat["left_after_semester"].any()
    assert (lat["shock"] < 0).any()
    assert (lat["n_failed_complete_data"] >= 2).any()
    assert set(dataset.students["trajectory"]) == {"improving", "stable", "declining"}


def test_label_prevalence_and_learnability_are_moderate(dataset, curriculum, target):
    outcomes = semester_outcomes(subject_outcomes(dataset.subjects, curriculum.grading), target)
    labels = next_semester_labels(outcomes, target)
    labelled = labels[labels["label_status"] == "labelled"]
    assert 0.10 <= labelled["label"].mean() <= 0.35
    # Not a deterministic function of the current SGPA: positives exist above any cut-off
    # that captures most negatives, and vice versa.
    joined = labelled.merge(
        outcomes[["student_key", "semester", "sgpa"]], on=["student_key", "semester"]
    ).dropna(subset=["sgpa"])
    middle = joined[joined["sgpa"].between(6.0, 7.0)]
    assert 0.05 < middle["label"].mean() < 0.95
