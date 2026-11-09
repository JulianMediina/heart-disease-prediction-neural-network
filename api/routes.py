"""Endpoints de la API (prefijo /api)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from .schemas import DatasetSummary, HealthResponse, PatientFeatures, PredictionResponse, risk_level_for
from .services import AppState

router = APIRouter(prefix="/api")


def _state(request: Request) -> AppState:
    return request.app.state.app_state


def _require(value: Any, detail: str | None) -> Any:
    if value is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail or "Recurso no disponible")
    return value


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    state = _state(request)
    return HealthResponse(status="ok", model_loaded=state.predictor is not None,
                          model_version=state.model_version, target_note=state.target_note)


@router.get("/features")
def features(request: Request) -> list[dict[str, Any]]:
    state = _state(request)
    return _require(state.features, state.errors.get("features"))


@router.get("/metrics")
def metrics(request: Request) -> dict[str, Any]:
    state = _state(request)
    return _require(state.metrics, state.errors.get("metrics"))


@router.get("/training-history")
def training_history(request: Request) -> dict[str, Any]:
    state = _state(request)
    return _require(state.history, state.errors.get("history"))


@router.get("/dataset/summary", response_model=DatasetSummary)
def dataset_summary(request: Request) -> dict[str, Any]:
    state = _state(request)
    return _require(state.dataset_summary, state.errors.get("dataset"))


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PatientFeatures, request: Request) -> PredictionResponse:
    # Función síncrona: FastAPI la ejecuta en un threadpool (la inferencia con torch bloquea).
    state = _state(request)
    predictor = _require(state.predictor, state.predictor_error)
    try:
        result = predictor.predict(payload.model_dump())
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    probability = float(result["probability"])
    return PredictionResponse(
        probability=probability,
        prediction=int(result["prediction"]),
        risk_level=risk_level_for(probability),
    )
