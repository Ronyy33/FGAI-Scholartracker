"""Optional prediction-history log (Phase 11), OFF by default.

When enabled (``SCHOLARTRACK_ENABLE_HISTORY=true``) each prediction request stores ONE summary
row: when, how many rows, how many were scored, the model and run, and counts per risk band.
No identifiers, names, marks or individual scores are stored, so the log cannot be used to look
up a student. Retention: rows older than ``retention_days`` are deleted on every write.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS prediction_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_utc TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('single', 'batch')),
    rows_received INTEGER NOT NULL,
    students_scored INTEGER NOT NULL,
    rows_rejected INTEGER NOT NULL,
    model TEXT NOT NULL,
    run_id TEXT NOT NULL,
    band_low INTEGER NOT NULL,
    band_moderate INTEGER NOT NULL,
    band_high INTEGER NOT NULL
)
"""


class PredictionHistory:
    def __init__(self, path: Path, retention_days: int = 90) -> None:
        self.path = path
        self.retention_days = retention_days
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as db:
            db.execute(SCHEMA)

    def record(
        self,
        source: str,
        rows_received: int,
        students_scored: int,
        rows_rejected: int,
        model: str,
        run_id: str,
        bands: dict[str, int],
    ) -> None:
        now = datetime.now(UTC)
        with sqlite3.connect(self.path) as db:
            db.execute(
                "INSERT INTO prediction_summaries (created_utc, source, rows_received, "
                "students_scored, rows_rejected, model, run_id, band_low, band_moderate, "
                "band_high) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    now.isoformat(timespec="seconds"),
                    source,
                    rows_received,
                    students_scored,
                    rows_rejected,
                    model,
                    run_id,
                    bands.get("Low", 0),
                    bands.get("Moderate", 0),
                    bands.get("High", 0),
                ),
            )
            cutoff = (now - timedelta(days=self.retention_days)).isoformat(timespec="seconds")
            db.execute("DELETE FROM prediction_summaries WHERE created_utc < ?", (cutoff,))

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with sqlite3.connect(self.path) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                "SELECT * FROM prediction_summaries ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
