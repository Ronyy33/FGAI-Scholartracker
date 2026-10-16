"""Scoring service: the evaluated models behind one function.

Every prediction — single JSON request, CSV upload or demo student — goes through
``ScoringService.score`` so that validation, adaptation, features, preprocessing and models are
exactly those used in training and evaluation. Nothing is trained or tuned here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scholartrack.data.curriculum import load_curriculum
from scholartrack.evaluation.calibration import apply_calibration
from scholartrack.explain import EXPLANATION_NOTE, explain, load_recommendation_rules, recommend
from scholartrack.features.build import build_sequences, semester_features
from scholartrack.features.preprocess import Preprocessor
from scholartrack.features.specs import load_feature_config
from scholartrack.models.artifacts import Predictor, load_network
from scholartrack.models.baselines import LogisticModel
from scholartrack.risk import load_risk_bands, risk_band, risk_score

LEARNED = ("logistic_regression", "mlp", "lstm")


class BundleError(RuntimeError):
    """The serving bundle is missing or does not match its recorded hashes."""


class ScoringService:
    def __init__(self, bundle_dir: Path) -> None:
        bundle_path = bundle_dir / "bundle.json"
        if not bundle_path.exists():
            raise BundleError(f"No serving bundle at {bundle_dir} (run export_serving_bundle.py)")
        self.bundle: dict[str, Any] = json.loads(bundle_path.read_text())
        for name, digest in self.bundle["files_sha256"].items():
            if hashlib.sha256((bundle_dir / name).read_bytes()).hexdigest() != digest:
                raise BundleError(f"{name} does not match bundle.json; refusing to serve it")
        self.preprocessor = Preprocessor.load(bundle_dir / "preprocessor.json")
        self.predictors: dict[str, Predictor] = {
            "logistic_regression": Predictor(
                "logistic_regression",
                "snapshot",
                LogisticModel.from_json((bundle_dir / "logistic_regression.json").read_text()),
            ),
        }
        for name, kind in (("mlp", "snapshot"), ("lstm", "sequence")):
            model, _ = load_network(bundle_dir, name)
            self.predictors[name] = Predictor(name, kind, model)
        self.curriculum = load_curriculum()
        self.features = load_feature_config()
        self.rules = load_recommendation_rules()
        self.bands = load_risk_bands()
        self.default_model: str = self.bundle["default_model"]

    @property
    def run_id(self) -> str:
        return self.bundle["run_id"]

    def score(
        self, semesters: pd.DataFrame, subjects: pd.DataFrame, with_explanations: bool = True
    ) -> list[dict[str, Any]]:
        """Score each student's latest semester (risk for the following semester)."""
        feats = semester_features(semesters, subjects, self.curriculum, self.features)
        matrix = self.preprocessor.transform(feats)
        seqs, lengths = build_sequences(feats, matrix, self.features.max_sequence_length)
        latest = feats.groupby("student_key")["semester"].idxmax().to_numpy()

        probs: dict[str, np.ndarray] = {}
        for name in LEARNED:
            raw = self.predictors[name].predict_proba(matrix[latest], seqs[latest], lengths[latest])
            probs[name] = apply_calibration(raw, self.bundle["models"][name]["calibration"])

        results = []
        default = self.default_model
        for i, row_index in enumerate(latest):
            row = feats.iloc[row_index]
            indicators = {f: _clean(row[f]) for f in self.features.snapshot_features}
            models = {}
            for name in LEARNED:
                p = float(probs[name][i])
                score = int(risk_score(np.array([p]))[0])
                models[name] = {
                    "probability": round(p, 4),
                    "risk_score": score,
                    "risk_band": str(risk_band(np.array([score]), self.bands)[0]),
                    "flagged": p >= self.bundle["models"][name]["threshold"],
                }
            result = {
                "student_key": row["student_key"],
                "semester": int(row["semester"]),
                "predicts_semester": int(row["semester"]) + 1,
                "default_model": default,
                **{
                    k: models[default][k]
                    for k in ("probability", "risk_score", "risk_band", "flagged")
                },
                "models": models,
                "indicators": indicators,
            }
            if with_explanations:
                result["explanation"] = explain(
                    self.predictors[default],
                    self.preprocessor,
                    matrix[row_index],
                    indicators,
                    seqs[row_index],
                    int(lengths[row_index]),
                )
                result["explanation_note"] = EXPLANATION_NOTE
                result["recommendations"] = recommend(indicators, self.rules)
            results.append(result)
        return results


def _clean(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return round(float(value), 3)
