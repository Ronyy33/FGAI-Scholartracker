"""Copy the evaluated run into the small serving bundle used by the API (models/serving/).

Usage:
    uv run python scripts/export_serving_bundle.py

The bundle holds exactly the models that were evaluated on the test set (not a retrained copy),
their frozen calibration/threshold choices, and hashes tying them to the synthetic dataset. It is
~100 KB and is committed so the deployed app serves the evaluated models. Only a run whose test
evaluation has been completed can be exported.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime

from scholartrack.evaluation.runner import latest_run_dir
from scholartrack.paths import PROJECT_ROOT, REPORTS_DIR

BUNDLE = PROJECT_ROOT / "models" / "serving"
FILES = ("preprocessor.json", "majority.json", "logistic_regression.json", "mlp.json", "mlp.pt",
         "lstm.json", "lstm.pt", "selection.json")


def main() -> None:
    run_dir = latest_run_dir()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if not manifest.get("test_partition_evaluated"):
        raise SystemExit("Refusing to export a run whose test evaluation has not been done.")
    BUNDLE.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in FILES:
        shutil.copy2(run_dir / name, BUNDLE / name)
        hashes[name] = hashlib.sha256((BUNDLE / name).read_bytes()).hexdigest()

    selection = json.loads((run_dir / "selection.json").read_text())
    evaluation = json.loads((REPORTS_DIR / "evaluation_metrics.json").read_text())
    chosen = selection["chosen_model"]
    bundle = {
        "bundle_version": 1,
        "exported_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_id": manifest["run_id"],
        "git_commit_of_training": manifest["git_commit"],
        "synthetic_data": True,
        "dataset_manifest_sha256": manifest["dataset"]["manifest_sha256"],
        "versions": manifest["versions"],
        "feature_names": manifest["feature_names"],
        "default_model": chosen,
        "models": {
            m: {"input": "sequence" if m == "lstm" else "snapshot",
                "calibration": s["calibration"], "threshold": s["threshold"],
                "test_roc_auc": evaluation["models"][m]["probability"]["roc_auc"],
                "test_pr_auc": evaluation["models"][m]["probability"]["pr_auc"],
                "test_brier": evaluation["models"][m]["probability"]["brier"]}
            for m, s in selection["models"].items()
        },
        "test_evaluated_utc": evaluation["evaluated_utc"],
        "files_sha256": hashes,
    }
    (BUNDLE / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
    size = sum((BUNDLE / f).stat().st_size for f in (*FILES, "bundle.json"))
    print(f"Exported run {manifest['run_id']} (default model: {chosen}) to "
          f"{BUNDLE.relative_to(PROJECT_ROOT)} — {size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
