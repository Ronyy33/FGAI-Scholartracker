"""Feature pipeline: correctness, no look-ahead, train/serve consistency, splits, preprocessing.

Synthetic data is generated in memory, so these tests also run in CI where data/ is empty.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scholartrack.data.adapter import adapt_college_frame
from scholartrack.data.loader import read_college_csv
from scholartrack.features.build import (
    build_modeling_table,
    build_sequences,
    entry_year,
    semester_features,
)
from scholartrack.features.preprocess import Preprocessor
from scholartrack.features.specs import load_feature_config
from scholartrack.features.split import split_students
from scholartrack.synthetic.generator import generate, load_synthetic_config, to_college_format
from scholartrack.targets import load_target_config
from tests.conftest import as_loaded, valid_row

KEYS = ["student_key", "semester"]


@pytest.fixture(scope="module")
def fconfig():
    return load_feature_config()


@pytest.fixture(scope="module")
def target():
    return load_target_config()


@pytest.fixture(scope="module")
def synthetic(curriculum, target):
    return generate(load_synthetic_config(), curriculum, target)


@pytest.fixture(scope="module")
def table(synthetic, curriculum, target, fconfig):
    return build_modeling_table(
        synthetic.semesters, synthetic.subjects, curriculum, target, fconfig
    )


def test_hand_computed_snapshot(curriculum, fconfig):
    """Two theory subjects (3 and 1 credits) and one termwork subject (1 credit)."""
    subjects = pd.DataFrame(
        [
            # code, credits, scheme, max, cie_max, ete_max, att, cie, ete, termwork, total
            ("A", 3, "theory", 100, 40, 60, 90.0, 30, 50, np.nan, 80),  # 80 % -> 9 points
            ("B", 1, "theory", 100, 40, 60, 60.0, 20, 15, np.nan, 35),  # 35 % -> fail, 0
            ("C", 1, "termwork", 50, np.nan, np.nan, 80.0, np.nan, np.nan, 30, 30),  # 60 % -> 7
        ],
        columns=[
            "subject_code",
            "credits",
            "scheme",
            "max_marks",
            "cie_max",
            "ete_max",
            "attendance_pct",
            "cie",
            "ete",
            "termwork",
            "total",
        ],
    )
    subjects.insert(0, "student_key", "S1")
    subjects.insert(1, "semester", 2)
    subjects["total_pct"] = subjects["total"] / subjects["max_marks"] * 100
    semesters = pd.DataFrame(
        {
            "student_key": ["S1"],
            "semester": [2],
            "prior_sgpa": [8.0],
            "study_hours_per_week": [np.nan],
        }
    )
    f = semester_features(semesters, subjects, curriculum, fconfig).iloc[0]
    assert f["sgpa_current"] == pytest.approx((9 * 3 + 0 * 1 + 7 * 1) / 5)
    assert f["sgpa_change"] == pytest.approx(f["sgpa_current"] - 8.0)
    assert f["credit_weighted_pct"] == pytest.approx((80 * 3 + 35 + 60) / 5)
    assert f["n_failed"] == 1
    assert f["share_subjects_below_50"] == pytest.approx(1 / 3)
    assert f["min_subject_pct"] == 35
    assert f["cie_mean_pct"] == pytest.approx((75 + 50) / 2)
    assert f["ete_mean_pct"] == pytest.approx((50 / 60 * 100 + 25) / 2)
    assert f["termwork_mean_pct"] == 60
    assert f["attendance_pct"] == pytest.approx((90 * 3 + 60 + 80) / 5)
    assert f["min_subject_attendance"] == 60
    assert f["share_subjects_attendance_below_75"] == pytest.approx(1 / 3)
    assert np.isnan(f["study_hours_per_week"])


def test_recomputed_summaries_match_reported_columns(synthetic, table):
    """On rows whose subject cells are all present, recomputed == reported (to rounding).

    The generator computes the reported columns *before* blanking cells, so a row with a blank
    subject cell is legitimately recomputed from fewer subjects; those rows are excluded.
    """
    subj = synthetic.subjects
    blank_att = subj["attendance_pct"].isna().groupby([subj["student_key"], subj["semester"]]).any()
    blank_total = subj["total"].isna().groupby([subj["student_key"], subj["semester"]]).any()
    reported = synthetic.semesters.set_index(KEYS)
    both = table.set_index(KEYS).join(
        reported[["credit_weighted_marks_pct", "overall_attendance_pct"]]
    )

    marks = both[~blank_total.reindex(both.index).to_numpy()].dropna(
        subset=["credit_weighted_marks_pct"]
    )
    assert len(marks) > 8000
    assert (
        (marks["credit_weighted_pct"] - marks["credit_weighted_marks_pct"]).abs() <= 0.05 + 1e-9
    ).all()
    att = both[~blank_att.reindex(both.index).to_numpy()].dropna(subset=["overall_attendance_pct"])
    assert len(att) > 7500
    assert ((att["attendance_pct"] - att["overall_attendance_pct"]).abs() <= 0.05 + 1e-9).all()


def test_features_never_look_ahead(synthetic, curriculum, fconfig):
    """Features (and sequences) for semester t are identical after deleting semesters > t."""
    full = semester_features(synthetic.semesters, synthetic.subjects, curriculum, fconfig)
    for t in (2, 5):
        sem = synthetic.semesters[synthetic.semesters["semester"] <= t]
        subj = synthetic.subjects[synthetic.subjects["semester"] <= t]
        truncated = semester_features(sem, subj, curriculum, fconfig)
        expected = full[full["semester"] <= t].reset_index(drop=True)
        pd.testing.assert_frame_equal(truncated, expected)

        cols = list(fconfig.snapshot_features)
        matrix_full = np.nan_to_num(full[cols].to_numpy(dtype=float))
        matrix_cut = np.nan_to_num(truncated[cols].to_numpy(dtype=float))
        seq_full, len_full = build_sequences(full, matrix_full, 8)
        seq_cut, len_cut = build_sequences(truncated, matrix_cut, 8)
        keep = (full["semester"] <= t).to_numpy()
        np.testing.assert_array_equal(seq_full[keep], seq_cut)
        np.testing.assert_array_equal(len_full[keep], len_cut)


def test_label_comes_from_the_next_semester_only(table):
    cols = [c for c in table.columns if c not in ("label", "label_status")]
    assert "label" not in cols
    # Rows whose next semester is unknown have no label at all.
    unlabelled = table["label_status"] != "labelled"
    assert table.loc[unlabelled, "label"].isna().all()


def test_train_serve_consistency_via_college_format(synthetic, curriculum, fconfig):
    """Features from the generator's tables == features from its exported CSV after adapting."""
    wide = to_college_format(synthetic, curriculum)[("2026-27", 5)]
    adapted = adapt_college_frame(read_college_csv(wide.to_csv(index=False).encode()), curriculum)
    served = semester_features(adapted.semesters, adapted.subjects, curriculum, fconfig)
    prn_to_key = dict(
        zip(synthetic.students["prn"], synthetic.students["student_key"], strict=True)
    )
    served["student_key"] = served["student_key"].map(
        {k: prn_to_key[p] for k, p in adapted.id_map.items()}
    )
    served = served.sort_values(KEYS, ignore_index=True)
    keys = set(served["student_key"])
    sem = synthetic.semesters[
        synthetic.semesters["student_key"].isin(keys) & (synthetic.semesters["semester"] == 5)
    ]
    subj = synthetic.subjects[
        synthetic.subjects["student_key"].isin(keys) & (synthetic.subjects["semester"] == 5)
    ]
    trained = semester_features(sem, subj, curriculum, fconfig)
    pd.testing.assert_frame_equal(served, trained, check_exact=False, atol=1e-9)


def test_single_reference_format_row_is_enough(curriculum, fconfig):
    adapted = adapt_college_frame(as_loaded([valid_row(curriculum, 1)]), curriculum)
    features = semester_features(adapted.semesters, adapted.subjects, curriculum, fconfig)
    assert features[list(fconfig.snapshot_features)].notna().all(axis=None)


def test_excluded_fields_are_not_features(fconfig, table):
    used = set(fconfig.snapshot_features)
    for field in ("gender", "age", "part_time_job", "commute_time_mins", "PRN", "student_name"):
        assert field not in used and field in fconfig.excluded
    assert not {"label", "label_status", "entry_year"} & used


def test_entry_year_from_academic_year():
    semesters = pd.DataFrame(
        {
            "student_key": ["A", "A", "B"],
            "semester": [1, 4, 5],
            "academic_year": ["2023-24", "2024-25", "2026-27"],
        }
    )
    assert entry_year(semesters).to_dict() == {"A": 2023, "B": 2024}


def test_split_is_by_student_reproducible_and_stratified(table, fconfig):
    years = table.groupby("student_key")["entry_year"].first()
    first = split_students(years, fconfig.split_fractions, fconfig.split_seed)
    again = split_students(
        years.sample(frac=1, random_state=3), fconfig.split_fractions, fconfig.split_seed
    )
    assert first.sort_index().equals(again.sort_index())  # independent of row order
    rows = table.assign(partition=table["student_key"].map(first))
    assert (rows.groupby("student_key")["partition"].nunique() == 1).all()
    shares = pd.crosstab(years, first, normalize="index")
    assert ((shares["train"] - 0.70).abs() < 0.01).all()
    assert ((shares["test"] - 0.15).abs() < 0.01).all()


def test_preprocessor_uses_training_statistics_only(table, fconfig):
    features, indicators = fconfig.snapshot_features, fconfig.missing_indicators
    train, other = table.iloc[:4000], table.iloc[4000:].copy()
    pre = Preprocessor.fit(train, features, indicators)
    assert pre.medians["prior_sgpa"] == pytest.approx(train["prior_sgpa"].median())
    other.loc[:, "prior_sgpa"] = 99.0  # changing non-training data cannot change the fit
    assert Preprocessor.fit(train, features, indicators) == pre

    x = pre.transform(train)
    assert x.shape == (len(train), len(features) + len(indicators))
    assert not np.isnan(x).any()
    scaled = x[:, : len(features)]
    np.testing.assert_allclose(scaled.mean(axis=0), 0, atol=1e-4)
    indicator_col = pre.output_names.index("prior_sgpa_missing")
    np.testing.assert_array_equal(x[:, indicator_col], train["prior_sgpa"].isna().to_numpy())


def test_preprocessor_json_round_trip_and_errors(table, fconfig, tmp_path):
    pre = Preprocessor.fit(table, fconfig.snapshot_features, fconfig.missing_indicators)
    path = tmp_path / "pre.json"
    pre.save(path)
    loaded = Preprocessor.load(path)
    np.testing.assert_array_equal(loaded.transform(table), pre.transform(table))
    with pytest.raises(ValueError, match="Missing feature columns"):
        pre.transform(table.drop(columns=["sgpa_current"]))


def test_sequences_are_prefixes_padded_with_zeros():
    table = pd.DataFrame({"student_key": ["B", "A", "A", "A"], "semester": [1, 3, 1, 2]})
    matrix = np.array([[9.0], [3.0], [1.0], [2.0]], dtype=np.float32)
    seqs, lengths = build_sequences(table, matrix, max_length=2)
    assert lengths.tolist() == [1, 2, 1, 2]
    np.testing.assert_array_equal(seqs[0, :, 0], [9, 0])  # B sem 1
    np.testing.assert_array_equal(seqs[2, :, 0], [1, 0])  # A sem 1
    np.testing.assert_array_equal(seqs[3, :, 0], [1, 2])  # A sem 2
    np.testing.assert_array_equal(seqs[1, :, 0], [2, 3])  # A sem 3, truncated to last two


def test_feature_config_rejects_bad_settings(tmp_path):
    good = load_feature_config().raw
    bad = dict(good, snapshot_features=[*good["snapshot_features"], "gender"])
    path = tmp_path / "features.yaml"
    import yaml

    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="Unknown features"):
        load_feature_config(path)


def test_data_dictionary_is_up_to_date():
    import sys

    from scholartrack.paths import PROJECT_ROOT

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from build_data_dictionary import OUTPUT, build

    assert OUTPUT.read_text(encoding="utf-8") == build(), (
        "docs/DATA_DICTIONARY.md is stale: run scripts/build_data_dictionary.py"
    )
