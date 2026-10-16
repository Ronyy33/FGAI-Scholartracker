"""Safe reading of college-format CSV files."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd


class CsvFormatError(ValueError):
    """The file cannot be interpreted as a college-format CSV at all."""


def _read_bytes(source: str | Path | bytes | io.IOBase) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, str | Path):
        return Path(source).read_bytes()
    data = source.read()
    return data.encode("utf-8") if isinstance(data, str) else data


def read_college_csv(
    source: str | Path | bytes | io.IOBase,
    *,
    max_rows: int | None = None,
    max_bytes: int | None = None,
) -> pd.DataFrame:
    """Read a college-format CSV without modifying the source.

    - ``utf-8-sig`` removes the byte-order mark some spreadsheet tools add, which would
      otherwise turn the first column name into ``"\\ufeffPRN"``.
    - Every column is read as text; :mod:`scholartrack.data.validation` converts numbers itself
      so that a stray value such as ``"12a"`` is reported rather than silently dropped.
    - Blank cells become missing values (``NaN``).
    - The header is checked for duplicate names *before* pandas reads the file, because pandas
      would otherwise silently rename the second copy (``a`` -> ``a.1``).
    """
    data = _read_bytes(source)
    if max_bytes is not None and len(data) > max_bytes:
        raise CsvFormatError(f"The file is {len(data)} bytes; the limit is {max_bytes}.")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CsvFormatError("The file is not UTF-8 encoded text.") from exc
    if not text.strip():
        raise CsvFormatError("The file is empty.")

    header = [h.strip() for h in next(csv.reader(io.StringIO(text)))]
    duplicated = sorted({h for h in header if header.count(h) > 1})
    if duplicated:
        raise CsvFormatError(f"Duplicate column names: {duplicated}")
    if any(h == "" for h in header):
        raise CsvFormatError("The header contains an empty column name.")

    try:
        frame = pd.read_csv(
            io.StringIO(text),
            dtype=str,
            keep_default_na=False,
            na_values=[""],
            skipinitialspace=True,
        )
    except pd.errors.ParserError as exc:
        raise CsvFormatError(f"The file is not a well-formed CSV: {exc}") from exc

    frame.columns = header
    if frame.empty:
        raise CsvFormatError("The file has a header but no data rows.")
    if max_rows is not None and len(frame) > max_rows:
        raise CsvFormatError(f"The file has {len(frame)} rows; the limit is {max_rows}.")

    for column in frame.columns:
        frame[column] = frame[column].str.strip()
    return frame.replace("", np.nan)
