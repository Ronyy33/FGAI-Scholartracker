"""Threshold-based metrics, threshold choice, Top-K risk capture and the student bootstrap."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    """Flag a student when p >= threshold. Undefined ratios are NaN, never silently 0."""
    y = np.asarray(y, dtype=int)
    flagged = np.asarray(p) >= threshold
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    tn = int((~flagged & (y == 0)).sum())

    def ratio(a: int, b: int) -> float:
        return a / b if b else float("nan")

    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision == precision and recall == recall and precision + recall > 0
        else float("nan")
    )
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": ratio(tp + tn, len(y)),
        "precision": precision,
        "recall": recall,
        "specificity": ratio(tn, tn + fp),
        "f1": f1,
        "flagged_share": float(flagged.mean()),
    }


def choose_threshold(y: np.ndarray, p: np.ndarray, criterion: str = "f1") -> float:
    """Threshold maximising the criterion over all distinct predicted probabilities."""
    if criterion != "f1":
        raise ValueError(f"Unsupported criterion {criterion!r}")
    # Exact predicted values: a rounded candidate could sit just above every prediction and
    # flag nobody (found with the constant majority baseline).
    candidates = np.unique(np.asarray(p, dtype=float))
    scores = [threshold_metrics(y, p, t)["f1"] for t in candidates]
    return float(candidates[int(np.nanargmax(scores))])


def top_k_capture(y: np.ndarray, p: np.ndarray, k: int) -> dict[str, float]:
    """Share of all actual positives found among the K highest-risk predictions.

    Top-K Risk Capture = positives in the top K / all positives × 100. It is NOT accuracy:
    it answers "if mentors can only meet K students, how many of those who will struggle do
    they reach?". Ties at the K-th score are broken by row order (stable sort).
    """
    y = np.asarray(y, dtype=int)
    order = np.argsort(-np.asarray(p), kind="stable")[:k]
    hits, total = int(y[order].sum()), int(y.sum())
    return {
        "k": int(k),
        "positives_in_top_k": hits,
        "total_positives": total,
        "capture_pct": 100 * hits / total if total else float("nan"),
        "precision_at_k_pct": 100 * hits / k if k else float("nan"),
    }


def student_bootstrap(
    groups: np.ndarray,
    statistic: Callable[[np.ndarray], dict[str, float]],
    n_resamples: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """95 % percentile intervals, resampling whole students (their rows are not independent)."""
    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(unique))]
    draws: dict[str, list[float]] = {}
    for _ in range(n_resamples):
        chosen = rng.integers(0, len(unique), size=len(unique))
        idx = np.concatenate([members[g] for g in chosen])
        for name, value in statistic(idx).items():
            draws.setdefault(name, []).append(value)
    return {
        name: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
        for name, v in draws.items()
    }
