"""Public pruning interface.

Use :func:`build_yolo_dependency_graph` and :func:`prune_yolo_channels` for
physical dependency-aware structured channel pruning.
"""

from .dependency_graph import (
    DependencyGraph,
    build_faster_rcnn_dependency_graph,
    build_yolo_dependency_graph,
)
from .cluster_selection import ClusterSpec, plan_low_importance_clusters
from .cluster_probe import make_keep_mask, run_filterwise_probe, run_structural_probe
from .importance import (
    compute_channel_importance,
    l1_filter_scores,
    minimum_weight_scores,
    rank_filters_by_minimum_weight,
    rank_output_channels,
)
from .unit_discovery import (
    DependencyOperation,
    DetectContract,
    FasterRCNNContract,
    PruningUnit,
    StructuralProbe,
    capture_detect_contract,
    capture_faster_rcnn_contract,
    discover_faster_rcnn_pruning_units,
    discover_pruning_units,
    requested_prune_count,
    serialize_dependency_group,
    validate_and_prune_unit,
    validate_and_prune_faster_rcnn_unit,
    validate_detect_contract,
    validate_faster_rcnn_contract,
)
from .yolo_pruner import prune_yolo_channels, synchronize_module_channel_metadata, validate_structural_reduction

__all__ = [
    "DependencyGraph",
    "build_faster_rcnn_dependency_graph",
    "build_yolo_dependency_graph",
    "ClusterSpec",
    "plan_low_importance_clusters",
    "make_keep_mask",
    "run_filterwise_probe",
    "run_structural_probe",
    "compute_channel_importance",
    "l1_filter_scores",
    "minimum_weight_scores",
    "rank_filters_by_minimum_weight",
    "rank_output_channels",
    "DependencyOperation",
    "DetectContract",
    "FasterRCNNContract",
    "PruningUnit",
    "StructuralProbe",
    "capture_detect_contract",
    "capture_faster_rcnn_contract",
    "discover_faster_rcnn_pruning_units",
    "discover_pruning_units",
    "requested_prune_count",
    "serialize_dependency_group",
    "validate_and_prune_unit",
    "validate_and_prune_faster_rcnn_unit",
    "validate_detect_contract",
    "validate_faster_rcnn_contract",
    "prune_yolo_channels",
    "synchronize_module_channel_metadata",
    "validate_structural_reduction",
]
