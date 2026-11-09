"""Esquemas Pydantic de entrada/salida de la API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RiskLevel = Literal["bajo", "moderado", "alto"]

# Umbrales de nivel de riesgo sobre la probabilidad predicha.
RISK_LOW_MAX = 0.33
RISK_MODERATE_MAX = 0.66


def risk_level_for(probability: float) -> RiskLevel:
    """bajo < 0.33 <= moderado < 0.66 <= alto."""
    if probability < RISK_LOW_MAX:
        return "bajo"
    if probability < RISK_MODERATE_MAX:
        return "moderado"
    return "alto"


class PatientFeatures(BaseModel):
    """Las 13 variables clínicas que espera el modelo.

    Los rangos coinciden con ``heart_nn.features.FEATURES`` (fuente de verdad, también
    servida en ``GET /api/features``): las variables continuas se limitan al rango
    observado en el dataset de entrenamiento para no extrapolar, y las categóricas
    reproducen la codificación del CSV (incluidos ``caa=4`` y ``thall=0``, que en el
    dataset de Kaggle representan datos faltantes). ``tests/test_api.py`` verifica que
    ambos catálogos no se desincronicen.
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,  # rechaza "1", true o 1.5 en lugar de enteros
        json_schema_extra={
            "example": {
                "age": 63, "sex": 1, "cp": 3, "trtbps": 145, "chol": 233, "fbs": 1,
                "restecg": 0, "thalachh": 150, "exng": 0, "oldpeak": 2.3, "slp": 0,
                "caa": 0, "thall": 1,
            }
        },
    )

    age: int = Field(..., ge=29, le=77, description="Edad en años")
    sex: int = Field(..., ge=0, le=1, description="Sexo (0 = mujer, 1 = hombre)")
    cp: int = Field(..., ge=0, le=3, description="Tipo de dolor torácico (0-3)")
    trtbps: int = Field(..., ge=94, le=200, description="Presión arterial en reposo (mm Hg)")
    chol: int = Field(..., ge=126, le=564, description="Colesterol sérico (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Glucemia en ayunas > 120 mg/dl (1 = sí)")
    restecg: int = Field(..., ge=0, le=2, description="Resultado del ECG en reposo (0-2)")
    thalachh: int = Field(..., ge=71, le=202, description="Frecuencia cardíaca máxima alcanzada")
    exng: int = Field(..., ge=0, le=1, description="Angina inducida por ejercicio (1 = sí)")
    oldpeak: float = Field(..., ge=0.0, le=6.2, description="Depresión del ST inducida por ejercicio")
    slp: int = Field(..., ge=0, le=2, description="Pendiente del segmento ST (0-2)")
    caa: int = Field(..., ge=0, le=4, description="Número de vasos principales (0-4)")
    thall: int = Field(..., ge=0, le=3, description="Resultado de la prueba de talio (0-3)")


class PredictionResponse(BaseModel):
    probability: float = Field(..., ge=0.0, le=1.0)
    prediction: Literal[0, 1]
    risk_level: RiskLevel


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    model_version: str | None
    # Advertencia sobre la semántica de la etiqueta (metadata.json -> target.note_es):
    # la clase 1 del dataset de Kaggle parece invertida respecto a UCI. Ver docs/AUDIT.md (DQ-03).
    target_note: str | None = None


class FeatureStats(BaseModel):
    mean: float
    std: float
    min: float
    max: float


class DatasetSummary(BaseModel):
    n_samples: int
    n_duplicates: int
    class_distribution: dict[str, int]
    features: dict[str, FeatureStats]
