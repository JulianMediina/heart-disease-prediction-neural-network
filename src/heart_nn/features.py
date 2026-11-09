"""Metadatos de las variables de entrada y pipeline de preprocesamiento.

Notas sobre la codificación de las variables categóricas
--------------------------------------------------------
El CSV es la versión "Kaggle" del dataset Cleveland (UCI). Sus códigos NO coinciden
con los del diccionario UCI original y el diccionario que suele acompañar al CSV
es incorrecto. Las etiquetas de ``FEATURES`` se derivaron comparando las
frecuencias del CSV con las del dataset UCI de 303 pacientes:

* ``cp``: 0 = asintomático (143 casos ~ UCI 144), 1 = angina atípica (50 ~ 50),
  2 = dolor no anginoso (87 ~ 86), 3 = angina típica (23 ~ 23).
* ``restecg``: 2 = anomalía ST-T (4 ~ 4); 1 = normal y 0 = hipertrofia ventricular
  izquierda, según la proporción de clases de cada grupo frente a UCI.
* ``slp``: 0 = descendente (21 ~ 21), 1 = plana (140 ~ 140), 2 = ascendente (142 ~ 142).
* ``thall``: 0 = dato faltante (2 ~ 2), 1 = defecto fijo (18 ~ 18),
  2 = normal (166 ~ 166), 3 = defecto reversible (117 ~ 117).
* ``caa``: 0-3 vasos; el valor 4 corresponde a datos faltantes en UCI (4 ~ 4+1 duplicado).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "output"

# La columna `output` del CSV de Kaggle está invertida respecto al dataset UCI Cleveland
# (ver README y docs/AUDIT.md, DQ-03): output=0 corresponde a pacientes CON enfermedad.
# El modelo se entrena sobre `enfermedad = 1 - output`, de modo que la clase positiva (1)
# es "enfermedad cardíaca".
LABEL = "enfermedad"


def disease_label(output: Any) -> Any:
    """Convierte la columna `output` del CSV en la etiqueta enfermedad (1 = enfermedad)."""
    return 1 - output

FEATURES: list[dict[str, Any]] = [
    {
        "name": "age",
        "label_es": "Edad",
        "description_es": "Edad del paciente en años.",
        "type": "numeric",
        "min": 29,
        "max": 77,
        "step": 1,
    },
    {
        "name": "sex",
        "label_es": "Sexo",
        "description_es": "Sexo biológico del paciente.",
        "type": "categorical",
        "min": 0,
        "max": 1,
        "step": 1,
        "options": [
            {"value": 0, "label_es": "Mujer"},
            {"value": 1, "label_es": "Hombre"},
        ],
    },
    {
        "name": "cp",
        "label_es": "Tipo de dolor torácico",
        "description_es": "Tipo de dolor en el pecho referido por el paciente.",
        "type": "categorical",
        "min": 0,
        "max": 3,
        "step": 1,
        "options": [
            {"value": 0, "label_es": "Asintomático"},
            {"value": 1, "label_es": "Angina atípica"},
            {"value": 2, "label_es": "Dolor no anginoso"},
            {"value": 3, "label_es": "Angina típica"},
        ],
    },
    {
        "name": "trtbps",
        "label_es": "Presión arterial en reposo",
        "description_es": "Presión arterial sistólica en reposo (mm Hg) al ingreso.",
        "type": "numeric",
        "min": 94,
        "max": 200,
        "step": 1,
    },
    {
        "name": "chol",
        "label_es": "Colesterol sérico",
        "description_es": "Colesterol sérico total (mg/dl).",
        "type": "numeric",
        "min": 126,
        "max": 564,
        "step": 1,
    },
    {
        "name": "fbs",
        "label_es": "Glucemia en ayunas > 120 mg/dl",
        "description_es": "Indica si la glucosa en ayunas supera 120 mg/dl.",
        "type": "categorical",
        "min": 0,
        "max": 1,
        "step": 1,
        "options": [
            {"value": 0, "label_es": "No"},
            {"value": 1, "label_es": "Sí"},
        ],
    },
    {
        "name": "restecg",
        "label_es": "Electrocardiograma en reposo",
        "description_es": "Resultado del electrocardiograma en reposo.",
        "type": "categorical",
        "min": 0,
        "max": 2,
        "step": 1,
        "options": [
            {"value": 0, "label_es": "Hipertrofia ventricular izquierda (criterios de Estes)"},
            {"value": 1, "label_es": "Normal"},
            {"value": 2, "label_es": "Anomalía de la onda ST-T"},
        ],
    },
    {
        "name": "thalachh",
        "label_es": "Frecuencia cardíaca máxima",
        "description_es": "Frecuencia cardíaca máxima alcanzada en la prueba de esfuerzo (lpm).",
        "type": "numeric",
        "min": 71,
        "max": 202,
        "step": 1,
    },
    {
        "name": "exng",
        "label_es": "Angina inducida por ejercicio",
        "description_es": "Indica si el ejercicio provocó angina.",
        "type": "categorical",
        "min": 0,
        "max": 1,
        "step": 1,
        "options": [
            {"value": 0, "label_es": "No"},
            {"value": 1, "label_es": "Sí"},
        ],
    },
    {
        "name": "oldpeak",
        "label_es": "Depresión del ST (oldpeak)",
        "description_es": (
            "Depresión del segmento ST inducida por el ejercicio respecto al reposo (mm)."
        ),
        "type": "numeric",
        "min": 0.0,
        "max": 6.2,
        "step": 0.1,
    },
    {
        "name": "slp",
        "label_es": "Pendiente del ST en esfuerzo",
        "description_es": "Pendiente del segmento ST en el pico del ejercicio.",
        "type": "categorical",
        "min": 0,
        "max": 2,
        "step": 1,
        "options": [
            {"value": 0, "label_es": "Descendente"},
            {"value": 1, "label_es": "Plana"},
            {"value": 2, "label_es": "Ascendente"},
        ],
    },
    {
        "name": "caa",
        "label_es": "Vasos principales coloreados",
        "description_es": "Número de vasos principales (0-3) coloreados por fluoroscopia.",
        "type": "categorical",
        "min": 0,
        "max": 4,
        "step": 1,
        "options": [
            {"value": 0, "label_es": "0 vasos"},
            {"value": 1, "label_es": "1 vaso"},
            {"value": 2, "label_es": "2 vasos"},
            {"value": 3, "label_es": "3 vasos"},
            {"value": 4, "label_es": "Desconocido (dato faltante en el origen)"},
        ],
    },
    {
        "name": "thall",
        "label_es": "Talasemia / gammagrafía con talio",
        "description_es": "Resultado de la prueba de perfusión con talio.",
        "type": "categorical",
        "min": 0,
        "max": 3,
        "step": 1,
        "options": [
            {"value": 0, "label_es": "Desconocido (dato faltante en el origen)"},
            {"value": 1, "label_es": "Defecto fijo"},
            {"value": 2, "label_es": "Normal"},
            {"value": 3, "label_es": "Defecto reversible"},
        ],
    },
]

FEATURE_NAMES: list[str] = [f["name"] for f in FEATURES]
FEATURES_BY_NAME: dict[str, dict[str, Any]] = {f["name"]: f for f in FEATURES}

# Variables nominales con más de dos niveles: candidatas a one-hot.
ONE_HOT_FEATURES: list[str] = ["cp", "restecg", "slp", "caa", "thall"]


def build_preprocessor(one_hot: bool = True) -> ColumnTransformer:
    """Crea el preprocesador (sin ajustar).

    Con ``one_hot=True`` las variables nominales multinivel se codifican one-hot
    (categorías fijadas a las del diccionario para que el número de columnas sea
    estable) y el resto se estandariza. Con ``one_hot=False`` se estandarizan las
    13 columnas tal cual (codificación ordinal, como el script original).
    """
    if not one_hot:
        return ColumnTransformer(
            [("num", StandardScaler(), FEATURE_NAMES)],
            remainder="drop",
            verbose_feature_names_out=False,
        )
    scaled = [n for n in FEATURE_NAMES if n not in ONE_HOT_FEATURES]
    categories = [
        [opt["value"] for opt in FEATURES_BY_NAME[n]["options"]] for n in ONE_HOT_FEATURES
    ]
    return ColumnTransformer(
        [
            ("num", StandardScaler(), scaled),
            (
                "cat",
                OneHotEncoder(categories=categories, handle_unknown="ignore", dtype=np.float32),
                ONE_HOT_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def validate_record(record: dict[str, Any]) -> dict[str, float]:
    """Valida un registro crudo y devuelve los valores como ``float``.

    Lanza ``ValueError`` si faltan variables, sobran variables, algún valor no es
    numérico, está fuera de rango o no es una categoría válida.
    """
    if not isinstance(record, dict):
        raise ValueError("El registro debe ser un diccionario {variable: valor}.")
    missing = [n for n in FEATURE_NAMES if n not in record]
    extra = sorted(k for k in record if k not in FEATURES_BY_NAME)
    if missing:
        raise ValueError(f"Faltan variables: {missing}")
    if extra:
        raise ValueError(f"Variables no reconocidas: {extra}")

    clean: dict[str, float] = {}
    for name in FEATURE_NAMES:
        spec = FEATURES_BY_NAME[name]
        raw = record[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float, np.integer, np.floating)):
            raise ValueError(f"'{name}' debe ser numérico, se recibió {raw!r}.")
        value = float(raw)
        if not np.isfinite(value):
            raise ValueError(f"'{name}' debe ser un número finito.")
        if spec["type"] == "categorical":
            allowed = {opt["value"] for opt in spec["options"]}
            if value not in allowed:
                raise ValueError(
                    f"'{name}'={raw!r} no es válido; valores permitidos: {sorted(allowed)}."
                )
        elif not spec["min"] <= value <= spec["max"]:
            raise ValueError(f"'{name}'={raw!r} fuera de rango [{spec['min']}, {spec['max']}].")
        clean[name] = value
    return clean
