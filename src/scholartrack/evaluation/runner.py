"""Reload a saved run and predict one partition with every model."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from scholartrack.data.curriculum import load_curriculum
from scholartrack.features.specs import load_feature_config
from scholartrack.models.artifacts import LoadedRun, load_run
from scholartrack.paths import PROJECT_ROOT
from scholartrack.targets import load_target_config
from scholartrack.training.data import load_dataset, prepare_data

ARTIFACTS = PROJECT_ROOT / "artifacts" / "next_semester_difficulty"


def latest_run_dir() -> Path:
    return ARTIFACTS / (ARTIFACTS / "LATEST").read_text().strip()


def predict_partition(run_dir: Path, partition: str) -> tuple[LoadedRun, pd.DataFrame]:
    """Rebuild the data exactly as in training and return per-row predictions of all models.

    Fails loudly if the dataset or the fitted preprocessor differ from the run's own, because
    then the predictions would not belong to the saved models.
    """
    run = load_run(run_dir)
    semesters, subjects, dataset_sha = load_dataset(PROJECT_ROOT / run.manifest["dataset"]["dir"])
    if dataset_sha != run.manifest["dataset"]["manifest_sha256"]:
        raise ValueError("The dataset changed since this run was trained; regenerate or retrain.")
    prepared = prepare_data(
        semesters, subjects, load_curriculum(), load_target_config(), load_feature_config()
    )
    if prepared.preprocessor != run.preprocessor:
        raise ValueError("Rebuilt preprocessor differs from the saved one; inputs changed.")
    part = prepared.partitions[partition]
    predictions = part.rows.assign(label=part.y)
    seconds = {}
    for name, predictor in run.predictors.items():
        start = time.perf_counter()
        predictions[name] = predictor.predict_proba(part.x, part.seq, part.lengths)
        seconds[name] = time.perf_counter() - start
    predictions.attrs["inference_seconds"] = seconds
    return run, predictions
