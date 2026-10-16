"""Runtime settings, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from scholartrack.paths import PROJECT_ROOT, REPORTS_DIR, SYNTHETIC_DIR


def _flag(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    bundle_dir: Path
    data_dir: Path
    reports_dir: Path
    frontend_dist: Path
    max_upload_bytes: int
    max_rows: int
    cors_origins: tuple[str, ...]
    enable_history: bool
    history_db: Path

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ.get
        origins = tuple(
            o.strip() for o in env("SCHOLARTRACK_CORS_ORIGINS", "").split(",") if o.strip()
        )
        return cls(
            bundle_dir=Path(env("SCHOLARTRACK_BUNDLE_DIR", PROJECT_ROOT / "models" / "serving")),
            data_dir=Path(env("SCHOLARTRACK_DATA_DIR", SYNTHETIC_DIR)),
            reports_dir=Path(env("SCHOLARTRACK_REPORTS_DIR", REPORTS_DIR)),
            frontend_dist=Path(
                env("SCHOLARTRACK_FRONTEND_DIST", PROJECT_ROOT / "frontend" / "dist")
            ),
            max_upload_bytes=int(env("SCHOLARTRACK_MAX_UPLOAD_BYTES", 2 * 1024 * 1024)),
            max_rows=int(env("SCHOLARTRACK_MAX_ROWS", 5000)),
            cors_origins=origins,
            enable_history=_flag("SCHOLARTRACK_ENABLE_HISTORY", False),
            history_db=Path(
                env("SCHOLARTRACK_HISTORY_DB", PROJECT_ROOT / "data" / "history.sqlite3")
            ),
        )
