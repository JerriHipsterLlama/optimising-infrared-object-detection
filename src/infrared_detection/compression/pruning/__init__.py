"""Public pruning interface.

Use :func:`build_yolo_dependency_graph` and :func:`prune_yolo_channels` for
physical structured pruning.  FCPTS calibration remains available through
``legacy`` and the YOLO-specific adapter through ``fcpts`` while its API is
consolidated.
"""

from .dependency_graph import DependencyGraph, build_yolo_dependency_graph
from .fcpts import CalibrationRunner, DifferentiablePruningMaskFn, calibrate_model, finalize_and_export
from .importance import compute_channel_importance
from .yolo_pruner import prune_yolo_channels, validate_structural_reduction

__all__ = [
    "DependencyGraph",
    "build_yolo_dependency_graph",
    "CalibrationRunner",
    "DifferentiablePruningMaskFn",
    "calibrate_model",
    "finalize_and_export",
    "compute_channel_importance",
    "prune_yolo_channels",
    "validate_structural_reduction",
]
