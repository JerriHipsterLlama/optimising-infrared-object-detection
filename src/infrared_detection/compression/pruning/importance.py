"""Channel importance criteria for structured pruning."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def minimum_weight_scores(module: nn.Conv2d) -> torch.Tensor:
    if not isinstance(module, nn.Conv2d):
        raise TypeError("MinimumWeight scoring requires torch.nn.Conv2d.")
    weight = module.weight.detach().to(dtype=torch.float32)
    return weight.square().mean(dim=tuple(range(1, weight.ndim)))


def rank_filters_by_minimum_weight(module: nn.Conv2d) -> tuple[tuple[int, float], ...]:
    scores = minimum_weight_scores(module).cpu().tolist()
    return tuple(sorted(enumerate(scores), key=lambda item: (item[1], item[0])))


def compute_channel_importance(
    model: nn.Module,
    graph: Any | None = None,
    criterion: str = "l1",
) -> dict[str, torch.Tensor]:
    """Return one importance score per output channel/filter."""

    del graph
    if criterion not in {"l1", "l2"}:
        raise ValueError("Supported channel importance criteria are 'l1' and 'l2'.")
    scores = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Torch-Pruning can leave the descriptive channel metadata stale
            # for branched modules after dependency propagation. The physical
            # tensor shape is the authoritative number of output filters.
            weight = module.weight.detach().reshape(module.weight.shape[0], -1)
            scores[name] = weight.abs().sum(dim=1) if criterion == "l1" else weight.square().sum(dim=1).sqrt()
    return scores
