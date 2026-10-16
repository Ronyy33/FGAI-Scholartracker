"""Phase 6 experiment: fit baselines, MLP and LSTM; select settings on validation only.

For each model family every grid setting is trained on the *training* students and scored on
the *validation* students. The setting with the lowest validation log loss is kept. For the two
networks, the chosen setting is then re-trained with further seeds to show how much results
vary from random initialisation alone. Nothing here reads the test partition.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd
import torch

from scholartrack.evaluation.metrics import probability_metrics
from scholartrack.models.artifacts import build_network, save_network
from scholartrack.models.baselines import LogisticModel, MajorityBaseline
from scholartrack.training.data import PreparedData
from scholartrack.training.loop import TrainSettings, predict_proba, set_seed, train_network


def _network_inputs(family: str, part) -> tuple[torch.Tensor, ...]:
    if family == "mlp":
        return (torch.as_tensor(part.x),)
    return (torch.as_tensor(part.seq), torch.as_tensor(part.lengths))


def _grid(family: str, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    grid = cfg[family]["grid"]
    keys = list(grid)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*grid.values())]


def fit_network(
    family: str, settings: dict[str, Any], prepared: PreparedData, cfg: dict[str, Any], seed: int
):
    tr, va = prepared.partitions["train"], prepared.partitions["validation"]
    spec = {"type": family, "n_inputs": int(tr.x.shape[1]), **settings}
    if family == "lstm":
        spec["num_layers"] = cfg["lstm"]["num_layers"]
    set_seed(seed)  # weight initialisation is random too
    model = build_network(spec)
    result = train_network(
        model,
        _network_inputs(family, tr),
        torch.as_tensor(tr.y, dtype=torch.float32),
        _network_inputs(family, va),
        torch.as_tensor(va.y, dtype=torch.float32),
        TrainSettings(**cfg[family]["training"]),
        seed,
    )
    p_val = predict_proba(model, _network_inputs(family, va))
    return model, spec, result, p_val


def run_experiment(prepared: PreparedData, cfg: dict[str, Any], out_dir) -> dict[str, Any]:
    tr, va = prepared.partitions["train"], prepared.partitions["validation"]
    seed, metric = cfg["seed"], cfg["selection_metric"]
    grid_rows: list[dict[str, Any]] = []
    chosen: dict[str, dict[str, Any]] = {}
    val_predictions = va.rows.assign(label=va.y)

    # Majority-class baseline
    majority = MajorityBaseline.fit(tr.y)
    (out_dir / "majority.json").write_text(majority.to_json() + "\n")
    p = majority.predict_proba(va.x)
    chosen["majority"] = {"settings": {}, "validation": probability_metrics(va.y, p)}
    val_predictions["majority"] = p

    # Logistic regression
    best = None
    for C, class_weight in itertools.product(
        cfg["logistic_regression"]["C"], cfg["logistic_regression"]["class_weight"]
    ):
        model = LogisticModel.fit(tr.x, tr.y, prepared.feature_names, C, class_weight, seed)
        p = model.predict_proba(va.x)
        scores = probability_metrics(va.y, p)
        settings = {"C": C, "class_weight": class_weight}
        grid_rows.append({"model": "logistic_regression", "settings": settings, **scores})
        if best is None or scores[metric] < best[2][metric]:
            best = (model, settings, scores, p)
    model, settings, scores, p = best
    (out_dir / "logistic_regression.json").write_text(model.to_json() + "\n")
    chosen["logistic_regression"] = {"settings": settings, "validation": scores}
    val_predictions["logistic_regression"] = p

    # Neural networks
    for family in ("mlp", "lstm"):
        best = None
        for settings in _grid(family, cfg):
            model, spec, result, p = fit_network(family, settings, prepared, cfg, seed)
            scores = probability_metrics(va.y, p)
            grid_rows.append(
                {
                    "model": family,
                    "settings": settings,
                    **scores,
                    "best_epoch": result.best_epoch,
                    "seconds": result.seconds,
                }
            )
            if best is None or scores[metric] < best[3][metric]:
                best = (model, spec, settings, scores, result, p)
        model, spec, settings, scores, result, p = best
        save_network(out_dir, family, model, spec)
        pd.DataFrame(result.history).to_csv(out_dir / f"{family}_history.csv", index=False)

        per_seed = [scores]
        for extra in range(1, cfg["seeds_for_chosen_networks"]):
            _, _, _, p_extra = fit_network(family, settings, prepared, cfg, seed + extra)
            per_seed.append(probability_metrics(va.y, p_extra))
        spread = {
            name: {
                "mean": float(np.mean([s[name] for s in per_seed])),
                "sd": float(np.std([s[name] for s in per_seed], ddof=1)),
            }
            for name in ("log_loss", "brier", "roc_auc", "pr_auc")
        }
        chosen[family] = {
            "settings": settings,
            "validation": scores,
            "best_epoch": result.best_epoch,
            "seconds": result.seconds,
            "seed_spread": spread,
            "seeds": [seed + i for i in range(len(per_seed))],
        }
        val_predictions[family] = p

    val_predictions.to_csv(out_dir / "validation_predictions.csv", index=False)
    return {"chosen": chosen, "grid": grid_rows}
