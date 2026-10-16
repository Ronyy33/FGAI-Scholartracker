"""Baseline models, persisted as readable JSON.

- ``MajorityBaseline`` predicts the training positive rate for everyone. It is the floor any
  useful model must beat; its ROC-AUC is 0.5 by construction.
- ``LogisticModel`` wraps a fitted scikit-learn logistic regression as plain coefficients, so
  the deployed artefact is a few numbers that can be read, explained and reloaded safely.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class MajorityBaseline:
    prevalence: float

    @classmethod
    def fit(cls, y: np.ndarray) -> MajorityBaseline:
        return cls(float(np.mean(y)))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.prevalence)

    def to_json(self) -> str:
        return json.dumps({"type": "majority", **asdict(self)}, indent=2)


@dataclass(frozen=True)
class LogisticModel:
    feature_names: tuple[str, ...]
    coef: tuple[float, ...]
    intercept: float
    C: float
    class_weight: str | None

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        C: float,
        class_weight: str | None,
        seed: int,
    ) -> LogisticModel:
        model = LogisticRegression(C=C, class_weight=class_weight, max_iter=5000, random_state=seed)
        model.fit(x, y)
        return cls(
            tuple(feature_names),
            tuple(float(c) for c in model.coef_[0]),
            float(model.intercept_[0]),
            float(C),
            class_weight,
        )

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float) @ np.asarray(self.coef) + self.intercept

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.decision_function(x)))

    def to_json(self) -> str:
        return json.dumps({"type": "logistic_regression", **asdict(self)}, indent=2)

    @classmethod
    def from_json(cls, text: str) -> LogisticModel:
        raw = json.loads(text)
        return cls(
            tuple(raw["feature_names"]),
            tuple(raw["coef"]),
            raw["intercept"],
            raw["C"],
            raw["class_weight"],
        )
