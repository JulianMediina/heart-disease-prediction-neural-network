"""Inferencia a partir de los artefactos guardados por ``heart_nn.train``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from heart_nn.features import FEATURE_NAMES, validate_record
from heart_nn.model import HeartClassifier


class HeartPredictor:
    """Envuelve preprocesador + red + umbral para predecir sobre registros crudos."""

    def __init__(self, model: HeartClassifier, preprocessor: Any, metadata: dict[str, Any]):
        if metadata.get("feature_names") != FEATURE_NAMES:
            raise ValueError("Los artefactos no corresponden a las variables esperadas.")
        self.model = model.eval()
        self.preprocessor = preprocessor
        self.metadata = metadata
        self.threshold = float(metadata["threshold"])

    @property
    def feature_names(self) -> list[str]:
        return list(self.metadata["feature_names"])

    @classmethod
    def from_artifacts(cls, directory: str | Path) -> HeartPredictor:
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        cfg = metadata["model_config"]
        model = HeartClassifier(cfg["input_dim"], cfg["hidden_sizes"], cfg["dropout"])
        state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        preprocessor = joblib.load(directory / "preprocessor.joblib")
        return cls(model, preprocessor, metadata)

    def predict_proba_frame(self, frame: pd.DataFrame) -> np.ndarray:
        """Probabilidades para un DataFrame ya validado con las 13 columnas crudas."""
        features = self.preprocessor.transform(frame[FEATURE_NAMES])
        tensor = torch.as_tensor(np.asarray(features, dtype=np.float32))
        return self.model.predict_proba(tensor).numpy()

    def predict_batch(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not isinstance(records, list) or not records:
            raise ValueError("Se espera una lista no vacía de registros.")
        clean = [validate_record(r) for r in records]
        proba = self.predict_proba_frame(pd.DataFrame(clean, columns=FEATURE_NAMES))
        return [{"probability": float(p), "prediction": int(p >= self.threshold)} for p in proba]

    def predict(self, record: dict[str, Any]) -> dict[str, Any]:
        return self.predict_batch([record])[0]
