"""Public pruning interface.

Use :func:`build_yolo_dependency_graph` and :func:`prune_yolo_channels` for
physical structured pruning.  FCPTS calibration remains available through
``legacy`` and the YOLO-specific adapter through ``fcpts`` while its API is
consolidated.
"""

from .dependency_graph import DependencyGraph, build_yolo_dependency_graph
from .cluster_selection import ClusterSpec, plan_low_importance_clusters
from .cluster_probe import make_keep_mask, run_filterwise_probe, run_structural_probe
from .fcpts import CalibrationRunner, DifferentiablePruningMaskFn, calibrate_model, finalize_and_export
from .importance import compute_channel_importance, minimum_weight_scores, rank_filters_by_minimum_weight
from .yolo_pruner import prune_yolo_channels, validate_structural_reduction

__all__ = [
    "DependencyGraph",
    "build_yolo_dependency_graph",
    "ClusterSpec",
    "plan_low_importance_clusters",
    "make_keep_mask",
    "run_filterwise_probe",
    "run_structural_probe",
    "CalibrationRunner",
    "DifferentiablePruningMaskFn",
    "calibrate_model",
    "finalize_and_export",
    "compute_channel_importance",
    "minimum_weight_scores",
    "rank_filters_by_minimum_weight",
    "prune_yolo_channels",
    "validate_structural_reduction",
]
