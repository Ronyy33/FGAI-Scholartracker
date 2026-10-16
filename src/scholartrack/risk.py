"""Academic Risk Score and risk bands.

``risk_score = round(100 × p)`` where ``p`` is the (calibrated) predicted probability of an
academic-difficulty semester next term. A score of 54 means an estimated 54 % chance — it is a
statement about one student, not about the model's accuracy.

Bands come from ``configs/risk_bands.yaml`` (demonstration defaults) and can change without
retraining.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from scholartrack.paths import CONFIG_DIR


@dataclass(frozen=True)
class RiskBand:
    name: str
    minimum: int
    maximum: int


def load_risk_bands(path: Path | str | None = None) -> tuple[RiskBand, ...]:
    path = Path(path) if path is not None else CONFIG_DIR / "risk_bands.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    bands = tuple(RiskBand(b["name"], int(b["min"]), int(b["max"])) for b in raw["bands"])
    expected = 0
    for band in sorted(bands, key=lambda b: b.minimum):
        if band.minimum != expected or band.maximum < band.minimum:
            raise ValueError(f"Risk bands must cover 0-100 without gaps or overlaps ({band})")
        expected = band.maximum + 1
    if expected != 101:
        raise ValueError("Risk bands must end at 100")
    return bands


def risk_score(probability: np.ndarray) -> np.ndarray:
    """0-100 integer score; halves round up (54.5 -> 55)."""
    p = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    return np.floor(100 * p + 0.5).astype(int)


def risk_band(score: np.ndarray, bands: tuple[RiskBand, ...]) -> np.ndarray:
    score = np.asarray(score)
    names = np.empty(score.shape, dtype=object)
    for band in bands:
        names[(score >= band.minimum) & (score <= band.maximum)] = band.name
    return names
