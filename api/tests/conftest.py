"""Fixtures: artefactos falsos y predictor simulado (no dependen del entrenamiento)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.services import FEATURE_NAMES
from api.settings import Settings

VALID_PAYLOAD: dict[str, Any] = {
    "age": 63, "sex": 1, "cp": 3, "trtbps": 145, "chol": 233, "fbs": 1, "restecg": 0,
    "thalachh": 150, "exng": 0, "oldpeak": 2.3, "slp": 0, "caa": 0, "thall": 1,
}

FAKE_METRICS = {
    "test": {"accuracy": 0.85, "precision": 0.86, "recall": 0.88, "f1": 0.87, "roc_auc": 0.91},
    "cv": {"accuracy": {"mean": 0.83, "std": 0.03}, "f1": {"mean": 0.85, "std": 0.03},
           "roc_auc": {"mean": 0.9, "std": 0.02}},
    "confusion_matrix": [[20, 5], [4, 32]],
    "roc_curve": {"fpr": [0.0, 0.5, 1.0], "tpr": [0.0, 0.9, 1.0]},
    "class_distribution": {"0": 138, "1": 164},
    "baselines": {"logistic_regression": {"accuracy": 0.84}},
    "n_samples": {"train": 193, "val": 48, "test": 61},
}
FAKE_HISTORY = {"train_loss": [0.7, 0.5, 0.4], "val_loss": [0.69, 0.55, 0.5], "best_epoch": 2}
FAKE_METADATA = {"feature_names": FEATURE_NAMES, "threshold": 0.5, "model_config": {},
                 "trained_at": "2026-11-09T00:00:00Z", "seed": 42, "version": "1.0.0",
                 "target": {"name": "output", "positive_class": 1,
                            "note_es": "Etiqueta posiblemente invertida."}}
FAKE_FEATURES = [{"name": n, "label_es": n, "description_es": n, "type": "int",
                  "min": 0, "max": 1, "step": 1} for n in FEATURE_NAMES]


class FakePredictor:
    def __init__(self, probability: float = 0.8) -> None:
        self.probability = probability
        self.calls: list[dict[str, Any]] = []

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(features)
        if features.get("chol") == 500:  # disparador de error de validación del modelo
            raise ValueError("entrada inválida")
        return {"probability": self.probability, "prediction": int(self.probability >= 0.5)}


def _write_artifacts(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in (("metrics.json", FAKE_METRICS), ("history.json", FAKE_HISTORY),
                          ("metadata.json", FAKE_METADATA)):
        (directory / name).write_text(json.dumps(content), encoding="utf-8")


def _write_csv(path: Path) -> None:
    header = ",".join(FEATURE_NAMES + ["output"])
    rows = [
        "63,1,3,145,233,1,0,150,0,2.3,0,0,1,1",
        "37,1,2,130,250,0,1,187,0,3.5,0,0,2,1",
        "37,1,2,130,250,0,1,187,0,3.5,0,0,2,1",
        "57,0,0,140,241,0,1,123,1,0.2,1,0,3,0",
    ]
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")


@pytest.fixture
def fake_predictor() -> FakePredictor:
    return FakePredictor()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    artifacts = tmp_path / "artifacts"
    _write_artifacts(artifacts)
    csv = tmp_path / "heart.csv"
    _write_csv(csv)
    return Settings(artifacts_dir=artifacts, data_path=csv, cors_origins="http://localhost:4200")


@pytest.fixture
def client(settings: Settings, fake_predictor: FakePredictor):
    app = create_app(settings, predictor_factory=lambda _p: fake_predictor,
                     features_loader=lambda: FAKE_FEATURES)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def degraded_client(tmp_path: Path):
    """API sin artefactos ni dataset ni paquete del modelo."""

    def failing_factory(_p: Path):
        raise FileNotFoundError("model.pt")

    def failing_features():
        raise ImportError("heart_nn")

    s = Settings(artifacts_dir=tmp_path / "nope", data_path=tmp_path / "nope.csv")
    app = create_app(s, predictor_factory=failing_factory, features_loader=failing_features)
    with TestClient(app) as c:
        yield c
