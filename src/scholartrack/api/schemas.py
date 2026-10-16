"""Pydantic request and response schemas (they also generate the OpenAPI docs at /api/docs)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

Scalar = str | int | float | bool | None


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Any = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class Health(BaseModel):
    status: str
    model_loaded: bool
    demo_data_loaded: bool
    run_id: str | None
    version: str


class PredictRequest(BaseModel):
    """One student's records in the college CSV format (one object per semester).

    The latest semester is scored; earlier semesters, if given, are used as history (LSTM).
    Example: ``{"records": [{"PRN": "X1", "semester": 5, "prior_sgpa": 7.1, ...}]}``
    """

    records: list[dict[str, Scalar]] = Field(min_length=1, max_length=8)


class ModelScore(BaseModel):
    probability: float
    risk_score: int
    risk_band: str
    flagged: bool


class Prediction(BaseModel):
    prn: str
    semester: int
    predicts_semester: int
    default_model: str
    probability: float
    risk_score: int = Field(ge=0, le=100)
    risk_band: str
    flagged: bool
    models: dict[str, ModelScore]
    indicators: dict[str, float | None]
    explanation: list[dict[str, Any]] = []
    explanation_note: str | None = None
    recommendations: dict[str, Any] | None = None


class Issue(BaseModel):
    severity: str
    code: str
    row: int | None
    column: str | None
    value: str | None
    message: str


class PredictResponse(BaseModel):
    synthetic_model: bool = True
    prediction: Prediction
    issues: list[Issue]


class BatchSummary(BaseModel):
    rows_received: int
    rows_used: int
    rows_rejected: int
    students_scored: int
    invalid_values: int
    warnings: int
    band_counts: dict[str, int]
    flagged: int


class BatchResponse(BaseModel):
    synthetic_model: bool = True
    summary: BatchSummary
    predictions: list[Prediction]
    issues: list[Issue]
