"""Channel importance criteria for structured pruning."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


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
