"""Entrenamiento del clasificador y generación de artefactos.

Uso::

    python -m heart_nn.train [--data data/heart.csv] [--out artifacts] [--loss bce|focal]
                             [--no-one-hot] [--seed 42]

Flujo:
1. Carga, validación de esquema y deduplicación del CSV.
2. Partición estratificada train/val/test (el test solo se usa al final).
3. Validación cruzada estratificada (k folds) sobre train+val: métricas medias ± std
   y probabilidades out-of-fold para fijar el umbral (índice de Youden).
4. Entrenamiento final: preprocesador ajustado SOLO con train, early stopping con
   val y restauración de los mejores pesos.
5. Evaluación única en test, baseline de regresión logística y escritura de artefactos.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from torch.utils.data import DataLoader, TensorDataset

from heart_nn import __version__
from heart_nn.config import DEFAULT_ARTIFACTS_DIR, DEFAULT_DATA_PATH, ModelConfig, TrainConfig
from heart_nn.data import load_dataset, split_data, split_xy
from heart_nn.evaluate import (
    compute_metrics,
    confusion,
    roc_points,
    summarize_folds,
    youden_threshold,
)
from heart_nn.features import FEATURE_NAMES, LABEL, TARGET, build_preprocessor
from heart_nn.losses import build_loss
from heart_nn.model import HeartClassifier

logger = logging.getLogger(__name__)


@dataclass
class TrainedPipeline:
    preprocessor: ColumnTransformer
    model: HeartClassifier
    history: dict[str, Any]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        features = to_tensor(self.preprocessor.transform(X[FEATURE_NAMES]))
        return self.model.predict_proba(features).numpy()


def set_seed(seed: int) -> None:
    """Fija todas las semillas relevantes y fuerza algoritmos deterministas."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array, dtype=np.float32))


def fit_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
) -> tuple[HeartClassifier, dict[str, Any]]:
    """Entrena el MLP con mini-lotes y early stopping sobre ``val``.

    Devuelve el modelo con los pesos de la mejor época (menor pérdida de validación).
    """
    set_seed(cfg.seed)
    xt, yt = to_tensor(X_train), to_tensor(y_train)
    xv, yv = to_tensor(X_val), to_tensor(y_val)
    generator = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(
        TensorDataset(xt, yt), batch_size=cfg.batch_size, shuffle=True, generator=generator
    )
    model = HeartClassifier(xt.shape[1], cfg.model.hidden_sizes, cfg.model.dropout)
    criterion = build_loss(cfg.loss, gamma=cfg.focal_gamma, alpha=cfg.focal_alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_loss, best_epoch, best_state = float("inf"), 0, copy.deepcopy(model.state_dict())

    for epoch in range(cfg.max_epochs):
        model.train()
        running = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
        train_losses.append(running / len(xt))

        model.eval()
        with torch.inference_mode():
            val_loss = criterion(model(xv), yv).item()
        val_losses.append(val_loss)

        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= cfg.patience:
            logger.debug("Early stopping en la época %d (mejor: %d).", epoch, best_epoch)
            break

    model.load_state_dict(best_state)
    model.eval()
    history = {"train_loss": train_losses, "val_loss": val_losses, "best_epoch": best_epoch}
    return model, history


def train_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: TrainConfig,
) -> TrainedPipeline:
    """Ajusta el preprocesador con ``train`` (nunca con val/test) y entrena la red."""
    preprocessor = build_preprocessor(one_hot=cfg.one_hot)
    Xt = preprocessor.fit_transform(X_train[FEATURE_NAMES])
    Xv = preprocessor.transform(X_val[FEATURE_NAMES])
    model, history = fit_network(Xt, y_train.to_numpy(), Xv, y_val.to_numpy(), cfg)
    return TrainedPipeline(preprocessor, model, history)


def inner_val_fraction(cfg: TrainConfig) -> float:
    return cfg.val_size / (1.0 - cfg.test_size)


def cross_validate(
    X: pd.DataFrame, y: pd.Series, cfg: TrainConfig
) -> tuple[list[dict[str, float]], np.ndarray]:
    """CV estratificada: en cada fold se separa una validación interna para early stopping.

    Devuelve las métricas por fold (umbral 0.5) y las probabilidades out-of-fold.
    """
    skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    oof = np.zeros(len(y), dtype=float)
    fold_metrics: list[dict[str, float]] = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_in, X_iv, y_in, y_iv = train_test_split(
            X_tr, y_tr, test_size=inner_val_fraction(cfg), stratify=y_tr, random_state=cfg.seed
        )
        pipe = train_pipeline(X_in, y_in, X_iv, y_iv, cfg)
        proba = pipe.predict_proba(X.iloc[te_idx])
        oof[te_idx] = proba
        metrics = compute_metrics(y.iloc[te_idx].to_numpy(), proba)
        fold_metrics.append(metrics)
        logger.info(
            "Fold %d/%d: acc=%.3f f1=%.3f auc=%.3f (mejor época %d)",
            fold + 1,
            cfg.cv_folds,
            metrics["accuracy"],
            metrics["f1"],
            metrics["roc_auc"],
            pipe.history["best_epoch"],
        )
    return fold_metrics, oof


def make_logistic_baseline(one_hot: bool, seed: int):
    return make_pipeline(
        build_preprocessor(one_hot=one_hot), LogisticRegression(max_iter=1000, random_state=seed)
    )


def logistic_cv(X: pd.DataFrame, y: pd.Series, cfg: TrainConfig) -> dict[str, dict[str, float]]:
    skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    folds = []
    for tr_idx, te_idx in skf.split(X, y):
        clf = make_logistic_baseline(cfg.one_hot, cfg.seed).fit(X.iloc[tr_idx], y.iloc[tr_idx])
        folds.append(
            compute_metrics(y.iloc[te_idx].to_numpy(), clf.predict_proba(X.iloc[te_idx])[:, 1])
        )
    return summarize_folds(folds)


def model_config_dict(cfg: TrainConfig, input_dim: int) -> dict[str, Any]:
    return {
        "architecture": "mlp",
        "input_dim": int(input_dim),
        "hidden_sizes": list(cfg.model.hidden_sizes),
        "dropout": cfg.model.dropout,
        "activation": "relu",
        "output": "logit",
        "loss": cfg.loss,
        "one_hot": cfg.one_hot,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(cfg: TrainConfig, data_path: Path, out_dir: Path) -> dict[str, Any]:
    """Ejecuta el flujo completo y escribe los artefactos en ``out_dir``."""
    set_seed(cfg.seed)
    df = load_dataset(data_path)
    split = split_data(df, cfg.test_size, cfg.val_size, cfg.seed)
    X_dev = pd.concat([split.X_train, split.X_val])
    y_dev = pd.concat([split.y_train, split.y_val])
    logger.info(
        "Particiones: train=%d val=%d test=%d",
        len(split.y_train),
        len(split.y_val),
        len(split.y_test),
    )

    logger.info("Validación cruzada estratificada (%d folds) sobre train+val...", cfg.cv_folds)
    fold_metrics, oof = cross_validate(X_dev, y_dev, cfg)
    cv_summary = summarize_folds(fold_metrics)
    threshold = youden_threshold(y_dev.to_numpy(), oof)
    logger.info("Umbral (Youden sobre predicciones out-of-fold): %.4f", threshold)

    logger.info("Entrenamiento final (early stopping con validación)...")
    pipe = train_pipeline(split.X_train, split.y_train, split.X_val, split.y_val, cfg)
    logger.info("Mejor época: %d", pipe.history["best_epoch"])

    # Única evaluación sobre test.
    test_proba = pipe.predict_proba(split.X_test)
    y_test = split.y_test.to_numpy()
    test_metrics = compute_metrics(y_test, test_proba, threshold)
    test_metrics_05 = compute_metrics(y_test, test_proba, 0.5)

    baseline = make_logistic_baseline(cfg.one_hot, cfg.seed).fit(X_dev, y_dev)
    baseline_metrics = compute_metrics(y_test, baseline.predict_proba(split.X_test)[:, 1], 0.5)

    input_dim = int(pipe.preprocessor.transform(split.X_train.iloc[:1][FEATURE_NAMES]).shape[1])
    metrics = {
        "test": test_metrics,
        "test_threshold_0_5": test_metrics_05,
        "cv": cv_summary,
        "cv_folds": cfg.cv_folds,
        "confusion_matrix": confusion(y_test, test_proba, threshold),
        "roc_curve": roc_points(y_test, test_proba),
        "class_distribution": {
            str(k): int(v) for k, v in split_xy(df)[1].value_counts().sort_index().items()
        },
        "baselines": {"logistic_regression": baseline_metrics},
        "baselines_cv": {"logistic_regression": logistic_cv(X_dev, y_dev, cfg)},
        "n_samples": {
            "train": len(split.y_train),
            "val": len(split.y_val),
            "test": len(split.y_test),
        },
        "threshold": threshold,
    }
    metadata = {
        "feature_names": list(FEATURE_NAMES),
        "threshold": threshold,
        "threshold_strategy": "youden_oof_cv",
        "model_config": model_config_dict(cfg, input_dim),
        "train_config": cfg.to_dict(),
        "target": {
            "name": LABEL,
            "source_column": TARGET,
            "definition": f"{LABEL} = 1 - {TARGET}",
            "positive_class": 1,
            "note_es": (
                "La predicción es la probabilidad de enfermedad cardíaca. La columna 'output' del "
                "CSV de Kaggle está invertida respecto al dataset UCI original, por lo que el "
                "modelo se entrena con enfermedad = 1 - output. Es un modelo educativo y no "
                "sustituye un diagnóstico médico."
            ),
        },
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": cfg.seed,
        "version": __version__,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(pipe.model.state_dict(), out_dir / "model.pt")
    joblib.dump(pipe.preprocessor, out_dir / "preprocessor.joblib")
    write_json(out_dir / "metadata.json", metadata)
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "history.json", pipe.history)
    log_summary(metrics)
    logger.info("Artefactos guardados en %s", out_dir)
    return metrics


def log_summary(metrics: dict[str, Any]) -> None:
    t, cv, lr = metrics["test"], metrics["cv"], metrics["baselines"]["logistic_regression"]
    logger.info(
        "TEST  acc=%.3f prec=%.3f rec=%.3f f1=%.3f auc=%.3f",
        t["accuracy"],
        t["precision"],
        t["recall"],
        t["f1"],
        t["roc_auc"],
    )
    logger.info(
        "CV    acc=%.3f±%.3f f1=%.3f±%.3f auc=%.3f±%.3f",
        cv["accuracy"]["mean"],
        cv["accuracy"]["std"],
        cv["f1"]["mean"],
        cv["f1"]["std"],
        cv["roc_auc"]["mean"],
        cv["roc_auc"]["std"],
    )
    logger.info("LOGREG acc=%.3f f1=%.3f auc=%.3f", lr["accuracy"], lr["f1"], lr["roc_auc"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Entrena el clasificador de enfermedad cardíaca.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Ruta del CSV.")
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_ARTIFACTS_DIR, help="Carpeta de artefactos."
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--loss", choices=["bce", "focal"], default=defaults.loss)
    parser.add_argument(
        "--no-one-hot", action="store_true", help="Codificación ordinal de categóricas."
    )
    parser.add_argument("--epochs", type=int, default=defaults.max_epochs)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--cv-folds", type=int, default=defaults.cv_folds)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = replace(
        TrainConfig(),
        seed=args.seed,
        loss=args.loss,
        one_hot=not args.no_one_hot,
        max_epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        cv_folds=args.cv_folds,
        model=ModelConfig(),
    )
    run(cfg, args.data, args.out)


if __name__ == "__main__":
    main()
