from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from heart_nn.config import DEFAULT_DATA_PATH, TrainConfig


@pytest.fixture(scope="session")
def data_path() -> Path:
    return DEFAULT_DATA_PATH


@pytest.fixture(scope="session")
def fast_cfg() -> TrainConfig:
    """Configuración reducida para que los tests sean rápidos."""
    return replace(TrainConfig(), max_epochs=40, patience=10, cv_folds=3)


@pytest.fixture
def sample_record() -> dict[str, float]:
    return {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trtbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalachh": 150,
        "exng": 0,
        "oldpeak": 2.3,
        "slp": 0,
        "caa": 0,
        "thall": 1,
    }


@pytest.fixture(scope="session")
def trained_artifacts(tmp_path_factory, data_path, fast_cfg) -> Path:
    from heart_nn.train import run

    out = tmp_path_factory.mktemp("artifacts")
    run(fast_cfg, data_path, out)
    return out
