"""Arquitectura del clasificador (MLP que devuelve logits)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class HeartClassifier(nn.Module):
    """MLP ``input -> hidden... -> 1`` con ReLU y Dropout.

    Devuelve *logits* (sin Sigmoid): la pérdida usa ``BCEWithLogitsLoss`` (más
    estable numéricamente) y la probabilidad se obtiene con ``torch.sigmoid``.
    """

    def __init__(
        self, input_dim: int, hidden_sizes: Sequence[int] = (64, 32), dropout: float = 0.3
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for size in hidden_sizes:
            layers += [nn.Linear(prev, size), nn.ReLU(), nn.Dropout(dropout)]
            prev = size
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    @torch.inference_mode()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self(x))
