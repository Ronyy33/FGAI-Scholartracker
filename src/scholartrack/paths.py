"""Well-known project locations, resolved relative to the repository root."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
REFERENCE_DIR = DATA_DIR / "reference"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
REPORTS_DIR = PROJECT_ROOT / "reports"

REFERENCE_CSV = REFERENCE_DIR / "scholartrack_synthetic_200_students_PRN_sorted.csv"
