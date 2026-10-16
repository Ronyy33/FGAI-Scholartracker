"""Evaluate a saved training run on one partition.

Usage:
    uv run python scripts/evaluate.py                         # latest run, validation partition
    uv run python scripts/evaluate.py --run <run_id>
    uv run python scripts/evaluate.py --partition test --final   # Phase 7: once, after model choice

The models are *reloaded from disk* and the data is rebuilt from the dataset files, so this also
checks that a saved run reproduces its own training-time validation predictions.

The test partition is guarded: it is only scored with ``--final``, and doing so is recorded in
the run's manifest (``test_partition_evaluated``), so it is visible afterwards whether, when and
how often the test set was looked at.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from scholartrack.data.curriculum import load_curriculum
from scholartrack.evaluation.metrics import probability_metrics
from scholartrack.features.specs import load_feature_config
from scholartrack.models.artifacts import load_run
from scholartrack.paths import PROJECT_ROOT
from scholartrack.targets import load_target_config
from scholartrack.training.data import load_dataset, prepare_data

ARTIFACTS = PROJECT_ROOT / "artifacts" / "next_semester_difficulty"


def evaluate(run_dir: Path, partition: str) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    rows = []
    for name, predictor in run.predictors.items():
        p = predictor.predict_proba(part.x, part.seq, part.lengths)
        predictions[name] = p
        rows.append({"model": name, **probability_metrics(part.y, p)})
    return pd.DataFrame(rows), predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default=None, help="run id (default: LATEST)")
    parser.add_argument("--partition", default="validation", choices=["validation", "test"])
    parser.add_argument("--final", action="store_true", help="required to score the test partition")
    args = parser.parse_args()

    run_id = args.run or (ARTIFACTS / "LATEST").read_text().strip()
    run_dir = ARTIFACTS / run_id
    if args.partition == "test" and not args.final:
        raise SystemExit(
            "Refusing to score the test partition without --final "
            "(use it once, after the model choice is frozen)."
        )

    metrics, predictions = evaluate(run_dir, args.partition)
    print(f"Run {run_id} — {args.partition} partition")
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if args.partition == "validation":
        saved = pd.read_csv(run_dir / "validation_predictions.csv")
        worst = max(
            float(np.abs(saved[m] - predictions[m]).max())
            for m in ("majority", "logistic_regression", "mlp", "lstm")
        )
        print(
            f"Reloaded models reproduce training-time validation predictions: "
            f"max |difference| = {worst:.2e}"
        )
        if worst > 1e-5:
            raise SystemExit("Saved run does NOT reproduce its own predictions.")
    else:
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["test_partition_evaluated"] = True
        manifest.setdefault("test_evaluations_utc", []).append(
            datetime.now(UTC).isoformat(timespec="seconds")
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        predictions.to_csv(run_dir / "test_predictions.csv", index=False)
        print("Recorded the test evaluation in manifest.json.")


if __name__ == "__main__":
    main()
