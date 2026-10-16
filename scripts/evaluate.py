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

from scholartrack.evaluation.metrics import probability_metrics
from scholartrack.evaluation.runner import ARTIFACTS, predict_partition

MODELS = ("majority", "logistic_regression", "mlp", "lstm")


def evaluate(run_dir: Path, partition: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, predictions = predict_partition(run_dir, partition)
    rows = [
        {"model": m, **probability_metrics(predictions["label"], predictions[m])} for m in MODELS
    ]
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
        worst = max(float(np.abs(saved[m] - predictions[m]).max()) for m in MODELS)
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
