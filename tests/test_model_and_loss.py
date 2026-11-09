from __future__ import annotations

import math

import pytest
import torch

from heart_nn.losses import FocalLoss, build_loss
from heart_nn.model import HeartClassifier


def manual_focal(logits, targets, gamma, alpha=None):
    out = []
    for z, y in zip(logits, targets, strict=True):
        p = 1 / (1 + math.exp(-z))
        p_t = p if y == 1 else 1 - p
        a_t = 1.0 if alpha is None else (alpha if y == 1 else 1 - alpha)
        out.append(-a_t * (1 - p_t) ** gamma * math.log(p_t))
    return out


@pytest.mark.parametrize("gamma,alpha", [(2.0, None), (1.0, 0.25), (0.0, None)])
def test_focal_matches_manual_formula(gamma, alpha):
    logits = [-2.0, -0.3, 0.0, 0.7, 3.0]
    targets = [0.0, 1.0, 1.0, 0.0, 1.0]
    expected = manual_focal(logits, targets, gamma, alpha)
    loss = FocalLoss(gamma=gamma, alpha=alpha, reduction="none")(
        torch.tensor(logits), torch.tensor(targets)
    )
    assert loss.tolist() == pytest.approx(expected, rel=1e-5)
    mean = FocalLoss(gamma=gamma, alpha=alpha)(torch.tensor(logits), torch.tensor(targets))
    assert mean.item() == pytest.approx(sum(expected) / len(expected), rel=1e-5)


def test_focal_is_per_sample_not_on_mean_bce():
    """El bug original aplicaba el factor focal a la BCE media; aquí no debe coincidir."""
    logits = torch.tensor([-3.0, 3.0, 0.1])
    targets = torch.tensor([0.0, 0.0, 1.0])
    bce_mean = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    buggy = (1 - torch.exp(-bce_mean)) ** 2 * bce_mean
    assert FocalLoss()(logits, targets).item() != pytest.approx(buggy.item(), rel=1e-3)


def test_focal_gamma_zero_equals_bce():
    logits = torch.randn(16)
    targets = (torch.rand(16) > 0.5).float()
    bce = build_loss("bce")(logits, targets)
    assert FocalLoss(gamma=0.0)(logits, targets).item() == pytest.approx(bce.item(), rel=1e-6)


def test_build_loss_rejects_unknown():
    with pytest.raises(ValueError):
        build_loss("mse")


def test_model_forward_shape():
    model = HeartClassifier(27, (64, 32), 0.3)
    out = model(torch.randn(8, 27))
    assert out.shape == (8,)
    proba = model.predict_proba(torch.randn(5, 27))
    assert proba.shape == (5,)
    assert torch.all((proba >= 0) & (proba <= 1))
