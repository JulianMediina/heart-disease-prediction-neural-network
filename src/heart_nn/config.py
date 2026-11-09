"""Configuración del entrenamiento y rutas por defecto."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "heart.csv"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

LossName = Literal["bce", "focal"]


@dataclass(frozen=True)
class ModelConfig:
    """Hiperparámetros de arquitectura del MLP."""

    hidden_sizes: tuple[int, ...] = (64, 32)
    dropout: float = 0.3


@dataclass(frozen=True)
class TrainConfig:
    """Hiperparámetros del experimento completo.

    Los valores por defecto (pérdida BCE, one-hot de categóricas) se eligieron
    con validación cruzada estratificada sobre el conjunto de desarrollo; ver
    ``python -m heart_nn.select`` y la sección de resultados del README.
    """

    seed: int = 42
    test_size: float = 0.15
    val_size: float = 0.15
    cv_folds: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 500
    patience: int = 30
    loss: LossName = "bce"
    focal_alpha: float | None = None
    focal_gamma: float = 2.0
    one_hot: bool = True
    model: ModelConfig = field(default_factory=ModelConfig)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["model"]["hidden_sizes"] = list(self.model.hidden_sizes)
        return data
