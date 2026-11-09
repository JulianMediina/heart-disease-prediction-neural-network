"""Selección de configuración (pérdida y codificación) por validación cruzada.

Uso::

    python -m heart_nn.select [--seeds 5]

Compara ``loss in {bce, focal}`` x ``one_hot in {True, False}`` con CV estratificada
de 5 folds repetida con varias semillas, usando solo train+val (el test no se toca).
El criterio principal es el ROC-AUC medio (independiente del umbral y de la escala de
la pérdida). Escribe ``artifacts/model_selection.json``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from heart_nn.config import DEFAULT_ARTIFACTS_DIR, DEFAULT_DATA_PATH, TrainConfig
from heart_nn.data import load_dataset, split_data
from heart_nn.evaluate import METRIC_NAMES
from heart_nn.train import cross_validate

logger = logging.getLogger(__name__)


def evaluate_config(
    X: pd.DataFrame, y: pd.Series, cfg: TrainConfig, seeds: list[int]
) -> dict[str, dict[str, float]]:
    folds: list[dict[str, float]] = []
    for seed in seeds:
        fold_metrics, _ = cross_validate(X, y, replace(cfg, seed=seed))
        folds.extend(fold_metrics)
    out = {}
    for name in METRIC_NAMES:
        values = np.array([m[name] for m in folds])
        out[name] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACTS_DIR / "model_selection.json")
    parser.add_argument("--seeds", type=int, default=5, help="Repeticiones de la CV.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("heart_nn.train").setLevel(logging.WARNING)

    base = TrainConfig()
    split = split_data(load_dataset(args.data), base.test_size, base.val_size, base.seed)
    X_dev = pd.concat([split.X_train, split.X_val])
    y_dev = pd.concat([split.y_train, split.y_val])
    seeds = list(range(args.seeds))

    results: list[dict[str, Any]] = []
    for loss, one_hot in itertools.product(["bce", "focal"], [True, False]):
        cfg = replace(base, loss=loss, one_hot=one_hot)
        summary = evaluate_config(X_dev, y_dev, cfg, seeds)
        results.append({"loss": loss, "one_hot": one_hot, "cv": summary})
        logger.info(
            "loss=%-5s one_hot=%-5s auc=%.4f±%.4f f1=%.4f±%.4f acc=%.4f±%.4f",
            loss,
            one_hot,
            summary["roc_auc"]["mean"],
            summary["roc_auc"]["std"],
            summary["f1"]["mean"],
            summary["f1"]["std"],
            summary["accuracy"]["mean"],
            summary["accuracy"]["std"],
        )
    best = max(results, key=lambda r: r["cv"]["roc_auc"]["mean"])
    logger.info(
        "Mejor configuración por ROC-AUC: loss=%s one_hot=%s", best["loss"], best["one_hot"]
    )
    payload = {
        "criterion": "roc_auc_mean",
        "cv_folds": base.cv_folds,
        "seeds": seeds,
        "results": results,
        "best": {"loss": best["loss"], "one_hot": best["one_hot"]},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
