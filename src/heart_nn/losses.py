"""Funciones de pérdida sobre logits."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812


class FocalLoss(nn.Module):
    """Focal loss binaria (Lin et al., 2017) calculada por muestra sobre logits.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    A diferencia de la versión original, el factor focal se aplica a la BCE de
    *cada* muestra antes de reducir, no a la media del lote.
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = None, reduction: str = "mean"):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"reduction inválida: {reduction}")
        if alpha is not None and not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha debe estar en [0, 1] o ser None.")
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)  # probabilidad asignada a la clase correcta
        loss = (1.0 - p_t) ** self.gamma * bce
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(name: str, gamma: float = 2.0, alpha: float | None = None) -> nn.Module:
    """Devuelve la pérdida configurada: ``"bce"`` o ``"focal"``."""
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "focal":
        return FocalLoss(gamma=gamma, alpha=alpha)
    raise ValueError(f"Pérdida desconocida: {name!r} (usa 'bce' o 'focal').")
