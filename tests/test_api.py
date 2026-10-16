"""API: endpoints, validation, limits, error format, privacy of logs, optional history."""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
import subprocess
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from scholartrack.api.app import create_app
from scholartrack.api.settings import Settings
from scholartrack.paths import PROJECT_ROOT, SYNTHETIC_DIR
from tests.conftest import valid_row


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    """The official synthetic dataset; regenerated (deterministically) if absent, e.g. in CI."""
    if (SYNTHETIC_DIR / "manifest.json").exists():
        return SYNTHETIC_DIR
    out = tmp_path_factory.mktemp("synthetic")
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_synthetic_data.py"),
            "--out-dir",
            str(out),
            "--no-report",
        ],
        check=True,
    )
    return out


@pytest.fixture(scope="session")
def settings(data_dir, tmp_path_factory):
    base = Settings.from_env()
    return dataclasses.replace(
        base,
        data_dir=data_dir,
        cors_origins=(),
        enable_history=False,
        frontend_dist=tmp_path_factory.mktemp("no_frontend"),
    )


@pytest.fixture(scope="session")
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


def _csv(rows) -> bytes:
    return pd.DataFrame(rows).to_csv(index=False).encode()


def test_health_and_model_info(client):
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["model_loaded"] and health["demo_data_loaded"]
    info = client.get("/api/model-info").json()
    assert info["synthetic_data"] is True
    assert info["default_model"] == "logistic_regression"
    assert [b["name"] for b in info["risk_bands"]] == ["Low", "Moderate", "High"]
    assert len(info["features"]) == 15


def test_metrics_endpoint(client):
    body = client.get("/api/metrics").json()
    assert body["selection"]["chosen_model"] == "logistic_regression"
    assert set(body["curves"]["roc"]) == {"logistic_regression", "mlp", "lstm"}


def test_overview_is_consistent(client):
    ov = client.get("/api/overview").json()
    assert ov["synthetic_data"] is True
    assert sum(b["count"] for b in ov["risk_bands"]) == ov["current_students_scored"] > 0
    assert 0.8 < ov["model"]["test_roc_auc"] < 1


def test_student_list_detail_and_not_found(client):
    listing = client.get("/api/students", params={"band": "High", "limit": 5}).json()
    assert listing["items"] and all(i["risk_band"] == "High" for i in listing["items"])
    prn = listing["items"][0]["prn"]
    detail = client.get(f"/api/students/{prn}").json()
    pred = detail["prediction"]
    assert pred["risk_band"] == "High" and len(pred["models"]) == 3
    assert pred["explanation"] and "not causes" in pred["explanation_note"]
    assert "decision support" in pred["recommendations"]["disclaimer"]
    assert [h["semester"] for h in detail["history"]] == sorted(
        h["semester"] for h in detail["history"]
    )
    graduated = client.get("/api/students/SYN19CE0001").json()
    assert graduated["prediction"] is None and graduated["message"]
    missing = client.get("/api/students/NOPE123")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "student_not_found"


def test_analytics_suppresses_small_groups(client):
    body = client.get("/api/analytics", params={"cohort": 2024, "semester": 5}).json()
    assert all(c["count"] >= body["min_group_size"] for c in body["attendance_vs_marks"]["cells"])
    assert all(t["n"] >= body["min_group_size"] for t in body["sgpa_trends"])
    assert body["subjects"]["semester"] == 5 and len(body["subjects"]["items"]) == 11


def test_single_prediction(client, curriculum):
    row = valid_row(curriculum, 1)
    response = client.post("/api/predict", json={"records": [row]})
    assert response.status_code == 200
    pred = response.json()["prediction"]
    assert pred["prn"] == row["PRN"] and pred["predicts_semester"] == 6
    assert 0 <= pred["risk_score"] <= 100
    assert pred["risk_score"] == pred["models"]["logistic_regression"]["risk_score"]


def test_single_prediction_reports_invalid_values_but_still_scores(client, curriculum):
    row = valid_row(curriculum, 2)
    row["prior_sgpa"] = 12.4
    body = client.post("/api/predict", json={"records": [row]}).json()
    assert any(i["code"] == "out_of_range" and i["column"] == "prior_sgpa" for i in body["issues"])
    assert body["prediction"]["indicators"]["prior_sgpa"] is None  # treated as missing


@pytest.mark.parametrize(
    ("payload", "status", "code"),
    [({"records": []}, 422, "invalid_request"), ({"wrong": 1}, 422, "invalid_request")],
)
def test_single_prediction_rejects_bad_bodies(client, payload, status, code):
    response = client.post("/api/predict", json=payload)
    assert response.status_code == status and response.json()["error"]["code"] == code


def test_single_prediction_requires_identifier_and_one_student(client, curriculum):
    row = valid_row(curriculum, 3)
    del row["PRN"]
    assert (
        client.post("/api/predict", json={"records": [row]}).json()["error"]["code"]
        == "invalid_file"
    )
    two = [valid_row(curriculum, 4), valid_row(curriculum, 5)]
    assert (
        client.post("/api/predict", json={"records": two}).json()["error"]["code"]
        == "multiple_students"
    )


def test_batch_prediction_and_privacy(client, curriculum, caplog):
    rows = [valid_row(curriculum, i) for i in range(6)]
    with caplog.at_level(logging.INFO, logger="scholartrack.api"):
        response = client.post(
            "/api/predict/batch", files={"file": ("upload.csv", _csv(rows), "text/csv")}
        )
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["students_scored"] == 6
    assert sorted(p["prn"] for p in body["predictions"]) == sorted(r["PRN"] for r in rows)
    assert "Synthetic Student" not in response.text  # names are never echoed
    log_text = " ".join(r.getMessage() for r in caplog.records)
    assert "6 rows" in log_text and "SYN0" not in log_text


@pytest.mark.parametrize(
    ("name", "content", "status", "code"),
    [
        ("upload.txt", b"PRN,semester\nA,5\n", 400, "unsupported_file"),
        ("upload.csv", b"", 400, "invalid_csv"),
        ("upload.csv", b"\xff\xfe\x00junk", 400, "invalid_csv"),
        ("upload.csv", b"PRN,name\nA,B\n", 422, "invalid_file"),
    ],
)
def test_batch_rejects_bad_files(client, name, content, status, code):
    response = client.post("/api/predict/batch", files={"file": (name, content, "text/csv")})
    assert response.status_code == status
    assert response.json()["error"]["code"] == code


def test_batch_limits(settings, curriculum):
    small = dataclasses.replace(settings, max_upload_bytes=2000, max_rows=3)
    with TestClient(create_app(small)) as c:
        big = _csv([valid_row(curriculum, i) for i in range(10)])
        assert (
            c.post("/api/predict/batch", files={"file": ("a.csv", big, "text/csv")}).status_code
            == 413
        )
        few_cols = b"PRN,semester\n" + b"".join(f"P{i},5\n".encode() for i in range(5))
        response = c.post("/api/predict/batch", files={"file": ("a.csv", few_cols, "text/csv")})
        assert response.status_code == 400 and "limit is 3" in response.json()["error"]["message"]


def test_history_is_off_by_default_and_stores_no_identifiers(
    client, settings, curriculum, tmp_path
):
    assert client.get("/api/history").json() == {"enabled": False, "items": []}
    with_history = dataclasses.replace(
        settings, enable_history=True, history_db=tmp_path / "h.sqlite3"
    )
    with TestClient(create_app(with_history)) as c:
        c.post(
            "/api/predict/batch",
            files={
                "file": ("a.csv", _csv([valid_row(curriculum, i) for i in range(3)]), "text/csv")
            },
        )
        items = c.get("/api/history").json()["items"]
    assert len(items) == 1 and items[0]["students_scored"] == 3
    assert "SYN" not in str(items[0])


def test_tampered_bundle_is_refused(settings, tmp_path):
    bundle = tmp_path / "bundle"
    shutil.copytree(settings.bundle_dir, bundle)
    (bundle / "logistic_regression.json").write_text('{"tampered": true}')
    with TestClient(create_app(dataclasses.replace(settings, bundle_dir=bundle))) as c:
        assert c.get("/health").json()["status"] == "degraded"
        response = c.get("/api/model-info")
        assert (
            response.status_code == 503 and response.json()["error"]["code"] == "model_unavailable"
        )


def test_documented_example_request_is_accepted(client):
    """examples/predict_request.json is used in the README and deployment guide."""
    body = json.loads((PROJECT_ROOT / "examples" / "predict_request.json").read_text())
    assert "student_name" not in body["records"][0]
    response = client.post("/api/predict", json=body)
    assert response.status_code == 200
    assert response.json()["prediction"]["prn"] == "EXAMPLE001"
