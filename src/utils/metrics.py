from __future__ import annotations

import torch


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def huber(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return torch.nn.functional.huber_loss(pred, target, delta=delta)
