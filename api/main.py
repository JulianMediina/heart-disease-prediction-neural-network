"""Punto de entrada: ``uvicorn api.main:app --port 8000``."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router
from .services import PredictorFactory, default_features_loader, default_predictor_factory, load_state
from .settings import Settings, get_settings

logging.basicConfig(level=logging.INFO)


def create_app(
    settings: Settings | None = None,
    predictor_factory: PredictorFactory = default_predictor_factory,
    features_loader: Callable[[], list[dict[str, Any]]] = default_features_loader,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # El predictor y los artefactos se cargan una única vez al arrancar.
        app.state.app_state = load_state(settings, predictor_factory, features_loader)
        yield

    app = FastAPI(
        title="Heart Disease Prediction API",
        description="API de inferencia del modelo de predicción de enfermedad cardíaca. "
        "Uso exclusivamente educativo: no es un dispositivo médico.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.include_router(router)
    return app


app = create_app()
