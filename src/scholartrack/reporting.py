"""Small helpers for writing Markdown reports without extra dependencies."""

from __future__ import annotations

import math

import pandas as pd


def fmt(value: object, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e9:
            return str(int(value))
        return f"{value:.{digits}f}"
    try:
        f = float(value)  # numpy scalars
    except (TypeError, ValueError):
        return str(value).replace("|", "\\|")
    return fmt(f, digits)


def md_table(frame: pd.DataFrame, digits: int = 2) -> str:
    header = "| " + " | ".join(str(c).replace("|", "\\|") for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = [
        "| " + " | ".join(fmt(v, digits) for v in row) + " |"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join([header, rule, *body])
