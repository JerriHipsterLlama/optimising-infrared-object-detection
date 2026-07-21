"""Execute individual structural channel-pruning probes."""

from __future__ import annotations

import torch
from torch import nn

from .cluster_selection import ClusterSpec
from .dependency_graph import build_yolo_dependency_graph
from .yolo_pruner import prune_yolo_channels


def make_keep_mask(output_channels: int, prune_indices: tuple[int, ...]) -> torch.Tensor:
    """Return a boolean mask that removes exactly ``prune_indices``."""

    keep_mask = torch.ones(output_channels, dtype=torch.bool)
    keep_mask[list(prune_indices)] = False
    return keep_mask


def run_structural_probe(model: nn.Module, example_input: torch.Tensor, spec: ClusterSpec) -> nn.Module:
    """Physically prune one candidate cluster through a fresh dependency graph."""

    graph = build_yolo_dependency_graph(model, example_input)
    module = graph.modules[spec.layer_name]
    output_channels = module.out_channels if isinstance(module, nn.Conv2d) else module.out_features
    keep_mask = make_keep_mask(output_channels, spec.prune_indices)
    if not bool(keep_mask.any()):
        raise ValueError(f"Probe for {spec.layer_name} removes every output channel.")
    return prune_yolo_channels(graph, {spec.layer_name: keep_mask})
