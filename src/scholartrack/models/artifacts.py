"""Saving and loading trained models behind one common interface.

Every model exposes ``predict_proba(x, seq, lengths)``: snapshot models read ``x``; the LSTM
reads ``seq`` and ``lengths``. Network weights are saved with ``torch.save(state_dict)`` and
loaded with ``weights_only=True``, which refuses arbitrary Python objects — a safety measure
for files that a server will load.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scholartrack.features.preprocess import Preprocessor
from scholartrack.models.baselines import LogisticModel, MajorityBaseline
from scholartrack.models.networks import MLP, LSTMClassifier
from scholartrack.training.loop import predict_proba

MODEL_NAMES = ("majority", "logistic_regression", "mlp", "lstm")


@dataclass
class Predictor:
    name: str
    input_kind: str  # "snapshot" or "sequence"
    model: Any

    def predict_proba(
        self, x: np.ndarray, seq: np.ndarray | None = None, lengths: np.ndarray | None = None
    ) -> np.ndarray:
        if self.name in ("majority", "logistic_regression"):
            return self.model.predict_proba(x)
        if self.input_kind == "snapshot":
            return predict_proba(self.model, (torch.as_tensor(x),))
        assert seq is not None and lengths is not None
        return predict_proba(self.model, (torch.as_tensor(seq), torch.as_tensor(lengths)))


def build_network(spec: dict[str, Any]) -> torch.nn.Module:
    if spec["type"] == "mlp":
        return MLP(spec["n_inputs"], list(spec["hidden_layers"]), spec["dropout"])
    if spec["type"] == "lstm":
        return LSTMClassifier(
            spec["n_inputs"], spec["hidden_size"], spec["num_layers"], spec["dropout"]
        )
    raise ValueError(f"Unknown network type {spec['type']!r}")


def save_network(directory: Path, name: str, model: torch.nn.Module, spec: dict[str, Any]) -> None:
    torch.save(model.state_dict(), directory / f"{name}.pt")
    (directory / f"{name}.json").write_text(json.dumps(spec, indent=2) + "\n")


def load_network(directory: Path, name: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    spec = json.loads((directory / f"{name}.json").read_text())
    model = build_network(spec)
    model.load_state_dict(torch.load(directory / f"{name}.pt", weights_only=True))
    model.eval()
    return model, spec


@dataclass
class LoadedRun:
    directory: Path
    manifest: dict[str, Any]
    preprocessor: Preprocessor
    predictors: dict[str, Predictor]


def load_run(directory: Path) -> LoadedRun:
    manifest = json.loads((directory / "manifest.json").read_text())
    preprocessor = Preprocessor.load(directory / "preprocessor.json")
    majority = json.loads((directory / "majority.json").read_text())
    predictors = {
        "majority": Predictor("majority", "snapshot", MajorityBaseline(majority["prevalence"])),
        "logistic_regression": Predictor(
            "logistic_regression",
            "snapshot",
            LogisticModel.from_json((directory / "logistic_regression.json").read_text()),
        ),
    }
    for name, kind in (("mlp", "snapshot"), ("lstm", "sequence")):
        model, _ = load_network(directory, name)
        predictors[name] = Predictor(name, kind, model)
    return LoadedRun(directory, manifest, preprocessor, predictors)
