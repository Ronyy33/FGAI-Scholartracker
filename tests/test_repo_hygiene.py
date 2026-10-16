"""Repository-hygiene checks.

The GitHub repository is public, so student-level data and secrets must never be committed.
These tests look at every file Git would commit (tracked files plus untracked files that are
not ignored) and fail if a data file or secret file is among them, or if the data folders stop
being ignored.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_SUFFIXES = {
    ".csv",
    ".tsv",
    ".xlsx",
    ".xls",
    ".parquet",
    ".feather",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".pkl",
    ".joblib",
    ".pt",
    ".pth",
    ".onnx",
}
# Only hand-made, fully synthetic fixtures may be committed.
ALLOWED_DATA_PREFIXES = ("tests/fixtures/",)

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not (REPO_ROOT / ".git").exists(),
    reason="not running inside a Git checkout",
)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )


def _committable_files() -> list[str]:
    result = _git("ls-files", "--cached", "--others", "--exclude-standard")
    assert result.returncode == 0, result.stderr
    return [line for line in result.stdout.splitlines() if line]


def test_no_data_files_outside_fixtures() -> None:
    offenders = [
        path
        for path in _committable_files()
        if Path(path).suffix.lower() in DATA_SUFFIXES and not path.startswith(ALLOWED_DATA_PREFIXES)
    ]
    assert offenders == [], f"Data/model files would be committed: {offenders}"


def test_no_env_files_committable() -> None:
    offenders = [
        path
        for path in _committable_files()
        if Path(path).name.startswith(".env") and Path(path).name != ".env.example"
    ]
    assert offenders == [], f"Environment/secret files would be committed: {offenders}"


@pytest.mark.parametrize(
    "path",
    [
        "data/reference/scholartrack_synthetic_200_students_PRN_sorted.csv",
        "data/external/uci_697/data.csv",
        "data/synthetic/students.csv",
        "data/processed/features.parquet",
        "data/anything_new.csv",
        "reports/predictions.csv",
        "artifacts/mlp/model.pt",
        ".env",
    ],
)
def test_sensitive_paths_are_ignored(path: str) -> None:
    result = _git("check-ignore", "--quiet", "--no-index", path)
    assert result.returncode == 0, f"{path} is NOT ignored by .gitignore"


def test_data_readme_is_not_ignored() -> None:
    result = _git("check-ignore", "--quiet", "--no-index", "data/README.md")
    assert result.returncode == 1, "data/README.md should be committed"
