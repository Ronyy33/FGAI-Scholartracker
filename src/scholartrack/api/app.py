"""ScholarTrack HTTP API (FastAPI).

Run locally:   uv run uvicorn scholartrack.api.app:app --reload
Docs:          http://localhost:8000/api/docs

The API never trains or tunes anything: it serves the evaluated bundle in models/serving/.
Logs contain counts only — never PRNs, names or individual scores.
"""

from __future__ import annotations

import io
import json
import logging
from contextlib import asynccontextmanager
from typing import Annotated, Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from scholartrack import __version__
from scholartrack.api.demo import DemoData, DemoUnavailable
from scholartrack.api.history import PredictionHistory
from scholartrack.api.schemas import (
    BatchResponse,
    ErrorResponse,
    Health,
    PredictRequest,
    PredictResponse,
)
from scholartrack.api.service import BundleError, ScoringService
from scholartrack.api.settings import Settings
from scholartrack.data.adapter import adapt_college_frame
from scholartrack.data.loader import CsvFormatError, read_college_csv
from scholartrack.data.validation import validate_college_frame
from scholartrack.features.specs import FEATURE_SPECS

log = logging.getLogger("scholartrack.api")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details: Any = None) -> None:
        self.status, self.code, self.message, self.details = status, code, message, details


def _error(status: int, code: str, message: str, details: Any = None) -> JSONResponse:
    body = {"error": {"code": code, "message": message, "details": details}}
    return JSONResponse(status_code=status, content=body)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    state: dict[str, Any] = {"service": None, "demo": None, "history": None, "load_error": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            state["service"] = ScoringService(settings.bundle_dir)
        except BundleError as exc:
            state["load_error"] = str(exc)
            log.error("Model bundle not loaded: %s", exc)
        if state["service"] is not None:
            try:
                state["demo"] = DemoData(settings.data_dir, state["service"])
                log.info("Demo data loaded: %d current students", len(state["demo"].current))
            except DemoUnavailable as exc:
                log.warning("Demo data unavailable: %s", exc)
        if settings.enable_history:
            state["history"] = PredictionHistory(settings.history_db)
        yield

    app = FastAPI(
        title="ScholarTrack API",
        version=__version__,
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        description="Academic Risk Score API. Models are trained on SYNTHETIC data; "
        "outputs are decision support, not decisions.",
        responses={
            400: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return _error(exc.status, exc.code, exc.message, exc.details)

    @app.exception_handler(HTTPException)
    async def _http_error(_: Request, exc: HTTPException):
        return _error(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        details = [{"location": list(e["loc"]), "message": e["msg"]} for e in exc.errors()]
        return _error(422, "invalid_request", "The request body is not valid.", details)

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception):
        log.exception("Unexpected error: %s", type(exc).__name__)
        return _error(500, "internal_error", "Unexpected server error.")

    def service() -> ScoringService:
        if state["service"] is None:
            raise ApiError(
                503, "model_unavailable", f"Model bundle not loaded: {state['load_error']}"
            )
        return state["service"]

    def demo() -> DemoData:
        if state["demo"] is None:
            raise ApiError(
                503, "demo_unavailable", "Synthetic demo data is not available on this server."
            )
        return state["demo"]

    def read_report(name: str) -> dict[str, Any]:
        path = settings.reports_dir / name
        if not path.exists():
            raise ApiError(503, "report_unavailable", f"{name} is not available.")
        return json.loads(path.read_text())

    @app.get("/health", response_model=Health, tags=["status"])
    @app.get("/api/health", response_model=Health, tags=["status"])
    def health() -> Health:
        svc = state["service"]
        return Health(
            status="ok" if svc else "degraded",
            model_loaded=svc is not None,
            demo_data_loaded=state["demo"] is not None,
            run_id=svc.run_id if svc else None,
            version=__version__,
        )

    @app.get("/api/model-info", tags=["model"])
    def model_info() -> dict[str, Any]:
        svc = service()
        b = svc.bundle
        return {
            "synthetic_data": True,
            "run_id": b["run_id"],
            "default_model": svc.default_model,
            "models": b["models"],
            "features": [
                {
                    "name": n,
                    "description": FEATURE_SPECS[n].description,
                    "unit": FEATURE_SPECS[n].unit,
                }
                for n in svc.features.snapshot_features
            ],
            "risk_bands": [{"name": r.name, "min": r.minimum, "max": r.maximum} for r in svc.bands],
            "target": "Probability that the NEXT semester has a failed subject or SGPA < 6.0 "
            "(illustrative thresholds; docs/TARGET_DEFINITION.md)",
            "dataset_manifest_sha256": b["dataset_manifest_sha256"],
            "versions": b["versions"],
            "limits": {
                "max_upload_bytes": settings.max_upload_bytes,
                "max_rows": settings.max_rows,
            },
            "prediction_history_enabled": settings.enable_history,
        }

    @app.get("/api/metrics", tags=["model"])
    def metrics() -> dict[str, Any]:
        evaluation = read_report("evaluation_metrics.json")
        curves = read_report("evaluation_curves.json")
        selection = read_report("model_selection.json")
        return {
            "evaluation": evaluation,
            "curves": curves,
            "selection": {
                "rule": selection["rule"],
                "chosen_model": selection["chosen_model"],
                "tied_within_tolerance": selection["tied_within_tolerance"],
                "models": {
                    m: {
                        "calibration": v["calibration"]["method"],
                        "threshold": v["threshold"],
                        "validation": v["validation"],
                    }
                    for m, v in selection["models"].items()
                },
            },
        }

    @app.get("/api/overview", tags=["demo"])
    def overview() -> dict[str, Any]:
        svc = service()
        out = demo().overview()
        evaluation = read_report("evaluation_metrics.json")
        chosen = evaluation["chosen_model"]
        out["model"] = {
            "run_id": svc.run_id,
            "default_model": svc.default_model,
            "test_roc_auc": evaluation["models"][chosen]["probability"]["roc_auc"],
            "test_pr_auc": evaluation["models"][chosen]["probability"]["pr_auc"],
            "test_positive_rate": evaluation["positive_rate"],
        }
        return out

    @app.get("/api/students", tags=["demo"])
    def students(
        q: str = Query("", max_length=20),
        band: str | None = None,
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ):
        return demo().list_students(q, band, limit, offset)

    @app.get("/api/students/{prn}", tags=["demo"])
    def student(prn: str) -> dict[str, Any]:
        detail = demo().student_detail(prn)
        if detail is None:
            raise ApiError(404, "student_not_found", "No synthetic student with that ID.")
        return detail

    @app.get("/api/analytics", tags=["demo"])
    def analytics(cohort: int | None = None, semester: int | None = Query(None, ge=1, le=8)):
        return demo().analytics(cohort, semester)

    def _score_frame(frame: pd.DataFrame, source: str) -> tuple[list, list, dict]:
        svc = service()
        validation = validate_college_frame(frame, svc.curriculum)
        report = validation.report
        issues = report.to_frame().astype(object).where(lambda d: d.notna(), None)
        issues = [dict(r) for r in issues.to_dict("records")]
        if report.has_fatal_error:
            raise ApiError(422, "invalid_file", "The data cannot be processed.", issues)
        adapted = adapt_college_frame(frame, svc.curriculum, validation)
        if adapted.semesters.empty:
            raise ApiError(422, "no_usable_rows", "No row could be used.", issues)
        results = svc.score(adapted.semesters, adapted.subjects)
        for r in results:
            r["prn"] = adapted.id_map[r.pop("student_key")]
        bands = pd.Series([r["risk_band"] for r in results]).value_counts().to_dict()
        summary = {
            "rows_received": int(len(frame)),
            "rows_used": int(len(adapted.semesters)),
            "rows_rejected": len(adapted.dropped_rows),
            "students_scored": len(results),
            "invalid_values": len(report.of("invalid")),
            "warnings": len(report.of("warning")),
            "band_counts": {b.name: int(bands.get(b.name, 0)) for b in svc.bands},
            "flagged": sum(r["flagged"] for r in results),
        }
        if state["history"] is not None:
            state["history"].record(
                source,
                summary["rows_received"],
                summary["students_scored"],
                summary["rows_rejected"],
                svc.default_model,
                svc.run_id,
                summary["band_counts"],
            )
        log.info(
            "%s prediction: %d rows, %d students scored, %d rows rejected",
            source,
            summary["rows_received"],
            summary["students_scored"],
            summary["rows_rejected"],
        )
        return results, issues, summary

    @app.post("/api/predict", response_model=PredictResponse, tags=["predict"])
    def predict(body: PredictRequest) -> dict[str, Any]:
        text = pd.DataFrame(body.records).to_csv(index=False)
        try:
            frame = read_college_csv(text.encode("utf-8"), max_rows=8)
        except CsvFormatError as exc:
            raise ApiError(400, "invalid_records", str(exc)) from exc
        if frame["PRN"].nunique() > 1 if "PRN" in frame else False:
            raise ApiError(
                400,
                "multiple_students",
                "All records must belong to one student; use /api/predict/batch.",
            )
        results, issues, _ = _score_frame(frame, "single")
        return {"prediction": results[0], "issues": issues}

    @app.post("/api/predict/batch", response_model=BatchResponse, tags=["predict"])
    async def predict_batch(
        file: Annotated[UploadFile, File(description="College-format CSV")],
    ) -> dict[str, Any]:
        name = (file.filename or "").lower()
        if not name.endswith(".csv"):
            raise ApiError(400, "unsupported_file", "Please upload a .csv file.")
        data = await file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise ApiError(
                413, "file_too_large", f"The file exceeds {settings.max_upload_bytes // 1024} KB."
            )
        try:
            frame = read_college_csv(
                io.BytesIO(data), max_rows=settings.max_rows, max_bytes=settings.max_upload_bytes
            )
        except CsvFormatError as exc:
            raise ApiError(400, "invalid_csv", str(exc)) from exc
        results, issues, summary = _score_frame(frame, "batch")
        return {"summary": summary, "predictions": results, "issues": issues}

    @app.get("/api/history", tags=["status"])
    def history(limit: int = Query(50, ge=1, le=500)) -> dict[str, Any]:
        if state["history"] is None:
            return {"enabled": False, "items": []}
        return {"enabled": True, "items": state["history"].recent(limit)}

    dist = settings.frontend_dist
    if (dist / "index.html").exists():
        if (dist / "assets").exists():
            app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith("api/"):
                raise ApiError(404, "not_found", "Unknown API endpoint.")
            target = dist / path
            if path and target.is_file() and dist in target.resolve().parents:
                return FileResponse(target)
            return FileResponse(dist / "index.html")

    return app


app = create_app()
