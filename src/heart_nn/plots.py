"""Gráficos a partir de los artefactos (no se ejecuta durante el entrenamiento).

Uso::

    python -m heart_nn.plots [--artifacts artifacts]

Guarda PNG en ``artifacts/figures``: curvas de pérdida, matriz de confusión,
curva ROC y distribución de clases. Requiere ``matplotlib`` (extra ``plots``).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from heart_nn.config import DEFAULT_ARTIFACTS_DIR  # noqa: E402

logger = logging.getLogger(__name__)


def plot_losses(history: dict[str, Any], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["train_loss"], label="Entrenamiento")
    ax.plot(history["val_loss"], label="Validación")
    ax.axvline(history["best_epoch"], color="grey", ls="--", label="Mejor época")
    ax.set(title="Pérdida por época", xlabel="Época", ylabel="Pérdida")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(cm: list[list[int]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", fontsize=14)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xlabel="Predicción",
        ylabel="Real",
        title="Matriz de confusión (test)",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_roc(roc: dict[str, list[float]], auc: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(roc["fpr"], roc["tpr"], label=f"MLP (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", ls="--")
    ax.set(
        xlabel="Tasa de falsos positivos",
        ylabel="Tasa de verdaderos positivos",
        title="Curva ROC (test)",
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_class_distribution(dist: dict[str, int], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    labels, values = list(dist.keys()), list(dist.values())
    bars = ax.bar(labels, values, color=["#4c72b0", "#dd8452"])
    ax.bar_label(bars)
    ax.set(
        title="Distribución de clases (tras deduplicar)",
        xlabel="enfermedad (1 = sí)",
        ylabel="Pacientes",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Genera gráficos a partir de los artefactos.")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    metrics = json.loads((args.artifacts / "metrics.json").read_text(encoding="utf-8"))
    history = json.loads((args.artifacts / "history.json").read_text(encoding="utf-8"))
    out = args.artifacts / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plot_losses(history, out / "loss.png")
    plot_confusion(metrics["confusion_matrix"], out / "confusion_matrix.png")
    plot_roc(metrics["roc_curve"], metrics["test"]["roc_auc"], out / "roc.png")
    plot_class_distribution(metrics["class_distribution"], out / "class_distribution.png")
    logger.info("Figuras guardadas en %s", out)


if __name__ == "__main__":
    main()
