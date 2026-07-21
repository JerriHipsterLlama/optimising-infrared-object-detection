"""Plan channel clusters for structural pruning probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Set

import torch


@dataclass(frozen=True)
class ClusterSpec:
    layer_name: str
    cluster_size: int
    prune_indices: tuple[int, ...]


def plan_low_importance_clusters(
    scores: Mapping[str, torch.Tensor],
    cluster_size: int,
    protected_layers: Set[str],
    max_clusters_per_layer: int = 1,
) -> list[ClusterSpec]:
    """Choose complete, low-importance channel clusters from unprotected layers."""

    clusters = []
    for layer_name in sorted(scores):
        if layer_name in protected_layers:
            continue
        ordered_indices = torch.argsort(scores[layer_name]).tolist()
        complete_count = len(ordered_indices) // cluster_size
        for group_index in range(min(complete_count, max_clusters_per_layer)):
            start = group_index * cluster_size
            clusters.append(
                ClusterSpec(layer_name, cluster_size, tuple(ordered_indices[start : start + cluster_size]))
            )
    return clusters
