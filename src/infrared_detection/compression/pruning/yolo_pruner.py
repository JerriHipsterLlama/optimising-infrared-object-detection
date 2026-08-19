"""YOLO structured-pruning operations backed by a dependency graph."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from .dependency_graph import DependencyGraph


def _find_module(graph: DependencyGraph, name: str) -> nn.Module:
    try:
        return graph.modules[name]
    except KeyError as exc:
        raise KeyError(f"Module is not a registered prunable layer: {name}") from exc


def synchronize_module_channel_metadata(model: nn.Module) -> nn.Module:
    """Synchronize declared channel counts with physically pruned tensor shapes."""

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            module.out_channels = int(module.weight.shape[0])
            module.in_channels = int(module.weight.shape[1]) * module.groups
        elif isinstance(module, nn.BatchNorm2d):
            module.num_features = int(module.weight.shape[0])
        elif isinstance(module, nn.Linear):
            module.out_features = int(module.weight.shape[0])
            module.in_features = int(module.weight.shape[1])
    return model


def prune_yolo_channels(graph: DependencyGraph, channel_masks: Mapping[str, torch.Tensor]) -> nn.Module:
    """Physically prune selected output channels and propagate dependencies."""

    try:
        import torch_pruning as tp
    except ImportError as exc:
        raise RuntimeError(
            "Structured pruning requires torch-pruning. Install the project dependency before pruning."
        ) from exc

    for name, keep_mask in channel_masks.items():
        module = _find_module(graph, name)
        if keep_mask.dtype != torch.bool or keep_mask.ndim != 1:
            raise ValueError(f"Channel mask for {name} must be a one-dimensional boolean tensor.")
        if int(keep_mask.sum()) <= 0 or int(keep_mask.sum()) >= keep_mask.numel():
            raise ValueError(f"Channel mask for {name} must remove at least one but not all channels.")
        prune_indices = (~keep_mask).nonzero(as_tuple=False).flatten().tolist()
        pruning_fn = tp.prune_conv_out_channels if isinstance(module, nn.Conv2d) else tp.prune_linear_out_channels
        group = graph.graph.get_pruning_group(module, pruning_fn, idxs=prune_indices)
        if not graph.graph.check_pruning_group(group):
            raise ValueError(f"Dependency graph rejected pruning group for {name}.")
        group.prune()
    return synchronize_module_channel_metadata(graph.model)


def validate_structural_reduction(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, float]:
    """Summarize physical parameter and serialized-size reductions."""

    before_params = float(before["parameter_count"])
    after_params = float(after["parameter_count"])
    before_bytes = float(before["serialized_bytes"])
    after_bytes = float(after["serialized_bytes"])
    if after_params >= before_params:
        raise ValueError("Structured pruning did not reduce parameter count.")
    return {
        "parameter_reduction": 1.0 - after_params / before_params,
        "serialized_reduction": 1.0 - after_bytes / before_bytes,
    }
