"""Carga, validación, deduplicación y partición del dataset."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from heart_nn.features import FEATURE_NAMES, FEATURES_BY_NAME, TARGET, disease_label

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS: list[str] = [*FEATURE_NAMES, TARGET]


@dataclass(frozen=True)
class DataSplit:
    """Particiones estratificadas train/val/test (DataFrames crudos, sin escalar)."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def validate_schema(df: pd.DataFrame) -> None:
    """Comprueba columnas, nulos, valores del objetivo y rangos/categorías conocidas."""
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Columnas inesperadas: {list(df.columns)}; se esperaba {EXPECTED_COLUMNS}"
        )
    if df.isna().any().any():
        raise ValueError("El dataset contiene valores nulos.")
    if not set(df[TARGET].unique()) <= {0, 1}:
        raise ValueError(f"La columna '{TARGET}' debe ser binaria (0/1).")
    for name in FEATURE_NAMES:
        spec = FEATURES_BY_NAME[name]
        col = df[name]
        if spec["type"] == "categorical":
            allowed = {opt["value"] for opt in spec["options"]}
            unknown = set(col.unique()) - allowed
            if unknown:
                raise ValueError(f"'{name}' contiene categorías desconocidas: {sorted(unknown)}")
        elif col.min() < spec["min"] or col.max() > spec["max"]:
            raise ValueError(f"'{name}' fuera del rango documentado [{spec['min']}, {spec['max']}]")


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas exactamente duplicadas (evita que la misma fila caiga en train y test)."""
    n_dup = int(df.duplicated().sum())
    if n_dup:
        logger.info("Se eliminan %d filas duplicadas.", n_dup)
    return df.drop_duplicates().reset_index(drop=True)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Lee el CSV, valida el esquema y elimina duplicados."""
    df = pd.read_csv(path)
    validate_schema(df)
    df = drop_duplicates(df)
    logger.info("Dataset cargado: %d filas, %d variables.", len(df), len(FEATURE_NAMES))
    return df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Devuelve X e y, con y = enfermedad (1 - output). Ver `features.LABEL`."""
    return df[FEATURE_NAMES].copy(), disease_label(df[TARGET].astype(int)).rename("enfermedad")


def split_data(df: pd.DataFrame, test_size: float, val_size: float, seed: int) -> DataSplit:
    """Partición estratificada train/val/test.

    ``test_size`` y ``val_size`` son fracciones del total. El test se separa primero
    y no se vuelve a tocar hasta la evaluación final.
    """
    X, y = split_xy(df)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    rel_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=rel_val, stratify=y_dev, random_state=seed
    )
    return DataSplit(X_train, y_train, X_val, y_val, X_test, y_test)
