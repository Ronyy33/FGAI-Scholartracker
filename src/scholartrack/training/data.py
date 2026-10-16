"""From the synthetic files to model-ready arrays, one call.

``prepare_data`` builds the modelling table, assigns partitions by student, fits the
preprocessor on **labelled training rows only**, and returns, for each partition, the snapshot
matrix (logistic regression, MLP), the padded sequences (LSTM) and the labels. Rows without a
label are kept in ``table`` (they can be scored) but are never part of a partition's arrays.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scholartrack.data.curriculum import Curriculum
from scholartrack.features.build import build_modeling_table, build_sequences
from scholartrack.features.preprocess import Preprocessor
from scholartrack.features.specs import FeatureConfig
from scholartrack.features.split import PARTITIONS, split_students
from scholartrack.targets import TargetConfig


@dataclass
class Partition:
    rows: pd.DataFrame  # student_key, semester, entry_year
    x: np.ndarray  # [n, f] snapshot features
    seq: np.ndarray  # [n, max_length, f] histories
    lengths: np.ndarray  # [n]
    y: np.ndarray  # [n] 0/1


@dataclass
class PreparedData:
    table: pd.DataFrame
    preprocessor: Preprocessor
    partitions: dict[str, Partition]

    @property
    def feature_names(self) -> list[str]:
        return self.preprocessor.output_names


def load_dataset(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Read the synthetic tables and check them against the generator's manifest."""
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found: run scripts/generate_synthetic_data.py first"
        )
    manifest = json.loads(manifest_path.read_text())
    for name in ("semesters.csv", "subjects.csv"):
        digest = hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
        if digest != manifest["files"][name]["sha256"]:
            raise ValueError(f"{name} does not match manifest.json (file changed?)")
    semesters = pd.read_csv(data_dir / "semesters.csv")
    subjects = pd.read_csv(data_dir / "subjects.csv")
    return semesters, subjects, hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def prepare_data(
    semesters: pd.DataFrame,
    subjects: pd.DataFrame,
    curriculum: Curriculum,
    target: TargetConfig,
    config: FeatureConfig,
) -> PreparedData:
    table = build_modeling_table(semesters, subjects, curriculum, target, config)
    years = table.groupby("student_key")["entry_year"].first()
    table["partition"] = table["student_key"].map(
        split_students(years, config.split_fractions, config.split_seed)
    )
    labelled = (table["label_status"] == "labelled").to_numpy()

    train_rows = table[labelled & (table["partition"] == "train").to_numpy()]
    preprocessor = Preprocessor.fit(train_rows, config.snapshot_features, config.missing_indicators)
    matrix = preprocessor.transform(table)
    sequences, lengths = build_sequences(table, matrix, config.max_sequence_length)

    partitions = {}
    for name in PARTITIONS:
        mask = labelled & (table["partition"] == name).to_numpy()
        partitions[name] = Partition(
            rows=table.loc[mask, ["student_key", "semester", "entry_year"]].reset_index(drop=True),
            x=matrix[mask],
            seq=sequences[mask],
            lengths=lengths[mask],
            y=table.loc[mask, "label"].to_numpy(dtype=int),
        )
    return PreparedData(table=table, preprocessor=preprocessor, partitions=partitions)
