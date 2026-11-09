"""Carga de artefactos, del predictor y cálculo del resumen del dataset."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .settings import Settings

logger = logging.getLogger("api")

FEATURE_NAMES = [
    "age", "sex", "cp", "trtbps", "chol", "fbs", "restecg",
    "thalachh", "exng", "oldpeak", "slp", "caa", "thall",
]
TARGET_CANDIDATES = ("output", "target")


class Predictor(Protocol):
    def predict(self, features: dict[str, Any]) -> dict[str, Any]: ...


PredictorFactory = Callable[[Path], Predictor]


@dataclass
class AppState:
    """Estado cargado una sola vez en el arranque (lifespan)."""

    predictor: Predictor | None = None
    predictor_error: str | None = None
    metadata: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    history: dict[str, Any] | None = None
    features: list[dict[str, Any]] | None = None
    dataset_summary: dict[str, Any] | None = None
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def target_note(self) -> str | None:
        target = (self.metadata or {}).get("target")
        note = target.get("note_es") if isinstance(target, dict) else None
        return str(note) if note else None

    @property
    def model_version(self) -> str | None:
        if self.predictor is None or not self.metadata:
            return None
        version = self.metadata.get("version")
        return str(version) if version is not None else None


def default_predictor_factory(artifacts_dir: Path) -> Predictor:
    from heart_nn.predict import HeartPredictor  # import diferido: torch es pesado

    return HeartPredictor.from_artifacts(artifacts_dir)


def default_features_loader() -> list[dict[str, Any]]:
    from heart_nn.features import FEATURES

    return [dict(f) for f in FEATURES]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def compute_dataset_summary(csv_path: Path) -> dict[str, Any]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    target = next((c for c in TARGET_CANDIDATES if c in df.columns), None)
    if target is None:
        raise ValueError(f"No se encontró la columna objetivo ({TARGET_CANDIDATES}) en {csv_path}")
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en {csv_path}: {missing}")

    # Misma etiqueta que el modelo: enfermedad = 1 - output (ver heart_nn.features.LABEL).
    counts = (1 - df[target]).value_counts()
    return {
        "n_samples": int(len(df)),
        "n_duplicates": int(df.duplicated().sum()),
        "class_distribution": {"0": int(counts.get(0, 0)), "1": int(counts.get(1, 0))},
        "features": {
            name: {
                "mean": float(df[name].mean()),
                "std": float(df[name].std()),
                "min": float(df[name].min()),
                "max": float(df[name].max()),
            }
            for name in FEATURE_NAMES
        },
    }


def load_state(
    settings: Settings,
    predictor_factory: PredictorFactory = default_predictor_factory,
    features_loader: Callable[[], list[dict[str, Any]]] = default_features_loader,
) -> AppState:
    """Carga todo lo necesario. Nunca lanza: los fallos se registran y se traducen en 503."""
    state = AppState()
    artifacts = Path(settings.artifacts_dir)

    # Los mensajes en ``state.errors`` se devuelven al cliente (503): no incluyen rutas
    # ni trazas internas. El detalle completo solo se registra en el log del servidor.
    for attr, filename in (("metadata", "metadata.json"), ("metrics", "metrics.json"),
                           ("history", "history.json")):
        try:
            setattr(state, attr, _read_json(artifacts / filename))
        except FileNotFoundError:
            state.errors[attr] = f"Artefacto {filename} no disponible; ejecute el entrenamiento."
            logger.warning("No existe %s", artifacts / filename)
        except (OSError, json.JSONDecodeError):
            state.errors[attr] = f"Artefacto {filename} no legible."
            logger.exception("No se pudo leer %s", artifacts / filename)

    try:
        state.predictor = predictor_factory(artifacts)
    except Exception:  # noqa: BLE001 - cualquier fallo deja la API en modo degradado
        state.predictor_error = "Modelo no disponible; ejecute el entrenamiento (heart-nn-train)."
        logger.exception("No se pudo cargar el predictor desde %s", artifacts)

    try:
        state.features = features_loader()
    except Exception:  # noqa: BLE001
        state.errors["features"] = "Catálogo de variables no disponible (paquete heart_nn)."
        logger.exception("No se pudo cargar heart_nn.features")

    try:
        state.dataset_summary = compute_dataset_summary(Path(settings.data_path))
    except Exception:  # noqa: BLE001
        state.errors["dataset"] = "Dataset no disponible."
        logger.exception("No se pudo calcular el resumen de %s", settings.data_path)

    return state
