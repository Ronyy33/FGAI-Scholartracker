"""Static evaluation figures (PNG) for the report.

Colours: the validated categorical order (blue, orange, aqua, yellow) on a light surface; every
model also gets its own line style, so identity never depends on colour alone, and every figure
has a legend. The numbers behind each figure are tabulated in the report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
SERIES = {
    "logistic_regression": "#2a78d6",
    "mlp": "#eb6834",
    "lstm": "#1baf7a",
    "majority": "#eda100",
}
STYLES = {"logistic_regression": "-", "mlp": "--", "lstm": "-.", "majority": ":"}
LABELS = {
    "logistic_regression": "Logistic regression",
    "mlp": "MLP",
    "lstm": "LSTM",
    "majority": "Majority baseline",
}
BLUE_RAMP = ["#86b6ef", "#2a78d6", "#104281"]  # ordinal steps 250 / 450 / 650


def _axes(title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, loc="left")
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)  # grid behind the data
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    return fig, ax


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def curve_plot(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    kind: str,
    baseline: float,
    path: Path,
    subtitle: str,
) -> None:
    """ROC (kind='roc') or precision-recall (kind='pr') curves for several models."""
    if kind == "roc":
        fig, ax = _axes(f"ROC curves — {subtitle}", "False positive rate", "True positive rate")
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=":", label="Random")
    else:
        fig, ax = _axes(f"Precision–recall curves — {subtitle}", "Recall", "Precision")
        ax.axhline(
            baseline,
            color=MUTED,
            linewidth=1,
            linestyle=":",
            label=f"Random (positive rate {baseline:.2f})",
        )
    for name, (x, y) in curves.items():
        ax.plot(x, y, color=SERIES[name], linestyle=STYLES[name], linewidth=2, label=LABELS[name])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(
        frameon=False, labelcolor=INK_2, loc="lower right" if kind == "roc" else "upper right"
    )
    _save(fig, path)


def calibration_plot(tables: dict[str, pd.DataFrame], path: Path, subtitle: str) -> None:
    fig, ax = _axes(
        f"Calibration — {subtitle}", "Mean predicted probability", "Observed difficulty rate"
    )
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=":", label="Perfect calibration")
    for name, table in tables.items():
        t = table.dropna()
        ax.plot(
            t["mean_predicted"],
            t["observed_rate"],
            color=SERIES[name],
            linestyle=STYLES[name],
            linewidth=2,
            marker="o",
            markersize=4,
            label=LABELS[name],
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, labelcolor=INK_2, loc="upper left")
    _save(fig, path)


def confusion_plot(tn: int, fp: int, fn: int, tp: int, path: Path, title: str) -> None:
    matrix = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.imshow(
        matrix,
        cmap=matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#104281"]),
    )
    for (i, j), value in np.ndenumerate(matrix):
        dark = value > matrix.max() * 0.55
        ax.text(
            j, i, f"{value}", ha="center", va="center", fontsize=13, color=SURFACE if dark else INK
        )
    ax.set_xticks([0, 1], ["Predicted: not at risk", "Predicted: at risk"], color=INK_2)
    ax.set_yticks([0, 1], ["Actual: no difficulty", "Actual: difficulty"], color=INK_2)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)
    _save(fig, path)


def risk_band_plot(bands: pd.DataFrame, path: Path, subtitle: str) -> None:
    """Students per risk band, annotated with the observed difficulty rate in each band."""
    fig, ax = _axes(f"Risk bands — {subtitle}", "Risk band", "Student-semesters")
    ax.grid(axis="x", visible=False)  # categories need no vertical rules
    bars = ax.bar(
        bands["band"],
        bands["count"],
        color=BLUE_RAMP[: len(bands)],
        width=0.6,
        edgecolor=SURFACE,
        linewidth=2,
    )
    for bar, rate in zip(bars, bands["observed_rate"], strict=True):
        ax.annotate(
            f"observed {rate * 100:.0f} %",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 3),
            textcoords="offset points",
            color=INK_2,
            fontsize=9,
        )
    _save(fig, path)
