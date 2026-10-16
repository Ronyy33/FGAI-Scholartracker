"""Preprocessing fitted on the training partition only, saved as plain JSON.

Steps, in order:
1. **Missing indicators** for selected features (1 if the value was missing, else 0).
2. **Median imputation**: a missing value is replaced by the *training* median.
3. **Standardisation**: (value − training mean) / training standard deviation.

Every number used in steps 2–3 is learned from training rows only, then frozen. Storing the
parameters as JSON (not a pickle) keeps the artefact human-readable and safe to load, and lets
the API apply exactly the same transformation as training did.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Preprocessor:
    features: tuple[str, ...]
    indicator_features: tuple[str, ...]
    medians: dict[str, float]
    means: dict[str, float]
    stds: dict[str, float]

    @classmethod
    def fit(
        cls, train: pd.DataFrame, features: tuple[str, ...], indicator_features: tuple[str, ...]
    ) -> Preprocessor:
        medians, means, stds = {}, {}, {}
        for name in features:
            values = train[name].astype(float)
            if values.notna().sum() == 0:
                raise ValueError(f"Feature '{name}' has no observed values in training data")
            median = float(values.median())
            filled = values.fillna(median)
            std = float(filled.std(ddof=0))
            medians[name], means[name] = median, float(filled.mean())
            stds[name] = std if std > 0 else 1.0
        return cls(tuple(features), tuple(indicator_features), medians, means, stds)

    @property
    def output_names(self) -> list[str]:
        return [*self.features, *(f"{name}_missing" for name in self.indicator_features)]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [f for f in self.features if f not in frame.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        scaled = [
            (
                (frame[name].astype(float).fillna(self.medians[name]) - self.means[name])
                / self.stds[name]
            ).to_numpy()
            for name in self.features
        ]
        indicators = [
            frame[name].isna().astype(float).to_numpy() for name in self.indicator_features
        ]
        return np.column_stack(scaled + indicators).astype(np.float32)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> Preprocessor:
        raw = json.loads(text)
        return cls(
            tuple(raw["features"]),
            tuple(raw["indicator_features"]),
            raw["medians"],
            raw["means"],
            raw["stds"],
        )

    def save(self, path: Path) -> None:
        path.write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Preprocessor:
        return cls.from_json(path.read_text(encoding="utf-8"))
