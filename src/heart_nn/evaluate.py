"""Métricas de clasificación y selección de umbral.

Ejecutable como ``python -m heart_nn.evaluate`` para re-evaluar los artefactos
guardados sobre la partición de test (reproducible gracias a la semilla).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

METRIC_NAMES: tuple[str, ...] = ("accuracy", "precision", "recall", "f1", "roc_auc")


def compute_metrics(
    y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Accuracy, precision, recall, F1 (con ``threshold``) y ROC-AUC."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    y_pred = (proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
    }


def confusion(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> list[list[int]]:
    """Matriz de confusión ``[[tn, fp], [fn, tp]]``."""
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    cm = confusion_matrix(np.asarray(y_true).astype(int), y_pred, labels=[0, 1])
    return cm.astype(int).tolist()


def roc_points(y_true: np.ndarray, proba: np.ndarray) -> dict[str, list[float]]:
    fpr, tpr, _ = roc_curve(np.asarray(y_true).astype(int), np.asarray(proba, dtype=float))
    return {"fpr": [round(float(v), 6) for v in fpr], "tpr": [round(float(v), 6) for v in tpr]}


def youden_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Umbral que maximiza el índice de Youden (TPR - FPR)."""
    fpr, tpr, thresholds = roc_curve(np.asarray(y_true).astype(int), np.asarray(proba, dtype=float))
    finite = np.isfinite(thresholds)
    j = (tpr - fpr)[finite]
    best = int(np.argmax(j))
    return float(np.clip(thresholds[finite][best], 0.0, 1.0))


def summarize_folds(fold_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Media y desviación estándar (muestral) de cada métrica entre folds."""
    out: dict[str, dict[str, float]] = {}
    for name in METRIC_NAMES:
        values = np.array([m[name] for m in fold_metrics], dtype=float)
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        out[name] = {"mean": float(values.mean()), "std": std}
    return out


def main(argv: list[str] | None = None) -> None:
    from heart_nn.config import DEFAULT_ARTIFACTS_DIR, DEFAULT_DATA_PATH
    from heart_nn.data import load_dataset, split_data
    from heart_nn.predict import HeartPredictor

    parser = argparse.ArgumentParser(description="Evalúa los artefactos sobre el conjunto de test.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    predictor = HeartPredictor.from_artifacts(args.artifacts)
    meta = json.loads((Path(args.artifacts) / "metadata.json").read_text(encoding="utf-8"))
    cfg = meta["train_config"]
    split = split_data(load_dataset(args.data), cfg["test_size"], cfg["val_size"], meta["seed"])
    proba = predictor.predict_proba_frame(split.X_test)
    metrics = compute_metrics(split.y_test.to_numpy(), proba, predictor.threshold)
    for name, value in metrics.items():
        logger.info("%-9s %.4f", name, value)
    logger.info(
        "Matriz de confusión [[tn, fp], [fn, tp]]: %s",
        confusion(split.y_test.to_numpy(), proba, predictor.threshold),
    )


if __name__ == "__main__":
    main()
