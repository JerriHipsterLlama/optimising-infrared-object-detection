"""Execute individual structural channel-pruning probes."""

from __future__ import annotations

import torch
from torch import nn

from .cluster_selection import ClusterSpec
from .dependency_graph import build_yolo_dependency_graph
from .yolo_pruner import prune_yolo_channels

_YOLO_DETECTION_HEAD_PREFIX = "model.22"


def make_keep_mask(output_channels: int, prune_indices: tuple[int, ...]) -> torch.Tensor:
    """Return a boolean mask that removes exactly ``prune_indices``."""

    keep_mask = torch.ones(output_channels, dtype=torch.bool)
    keep_mask[list(prune_indices)] = False
    return keep_mask


def run_structural_probe(model: nn.Module, example_input: torch.Tensor, spec: ClusterSpec) -> nn.Module:
    """Physically prune one candidate cluster through a fresh dependency graph."""

    return run_filterwise_probe(model, example_input, spec.layer_name, spec.prune_indices)


def run_filterwise_probe(
    model: nn.Module,
    example_input: torch.Tensor,
    layer_name: str,
    prune_indices: tuple[int, ...],
) -> nn.Module:
    """Physically remove an explicit set of output filters from one layer."""

    if layer_name == _YOLO_DETECTION_HEAD_PREFIX or layer_name.startswith(
        f"{_YOLO_DETECTION_HEAD_PREFIX}."
    ):
        raise ValueError(f"Filter-wise probes cannot target detection head module {layer_name}.")
    graph = build_yolo_dependency_graph(model, example_input)
    module = graph.modules[layer_name]
    output_channels = int(module.weight.shape[0])
    keep_mask = make_keep_mask(output_channels, prune_indices)
    if not bool(keep_mask.any()):
        raise ValueError(f"Probe for {layer_name} removes every output channel.")
    return prune_yolo_channels(graph, {layer_name: keep_mask})
