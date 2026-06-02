# References: 
#  [1] C. Gamanayake, L. Jayasinghe, B. K. K. Ng, and C. Yuen, “Cluster Pruning: An Efficient Filter Pruning Method for Edge AI Vision Applications,” IEEE Journal on Selected Topics in Signal Processing, vol. 14, no. 4, pp. 802–816, May 2020, doi: 10.1109/JSTSP.2020.2971418.
#  
# Takes a simple ONNX model and applies cluster pruning to the backbone conv layer,
# while ensuring the head conv layer is protected from pruning. The test verifies 
# that the correct channels are pruned and that the protected head conv remains intact.

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from dataclasses import dataclass, field
import numpy as np
import onnx
from onnx import numpy_helper
import os
import time
import onnxruntime as ort

# Guesstimate of protected keywords in layer names that should not be pruned
DEFAULT_PROTECTED_KEYWORDS: Tuple[str, ...] = (
	"head",
	"cls",
	"class",
	"detect",
	"detection",
	"bbox",
	"box",
	"rpn",
	"roi",
	"proposal",
	"mask",
)

DEFAULT_PROTECTED_OP_TYPES: Tuple[str, ...] = (
	"NonMaxSuppression",
	"Softmax",
	"Sigmoid",
	"Reshape",
	"Transpose",
	"Concat",
)

@dataclass(slots=True)
class StructuredClusterPruningConfig:
    """Configuration for cluster pruning."""

    onnx_model_path: Path
    output_model_path: Path
    pruning_percentage: float = 0.2
    cluster_size: int = 4
    protected_keywords: Tuple[str, ...] = field(default_factory=lambda: DEFAULT_PROTECTED_KEYWORDS)
    protected_op_types: Tuple[str, ...] = field(default_factory=lambda: DEFAULT_PROTECTED_OP_TYPES)

# --- ONNX Model Handling ---
def load_onnx_model(model_path: Path) -> onnx.ModelProto:
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model file not found: {model_path}")
    return onnx.load(str(model_path))

def save_onnx_model(model: onnx.ModelProto, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(model_path))

# --- NumPy Helper Functions ---
def build_initializer_map(model: onnx.ModelProto) -> Dict[str, np.ndarray]:
    return { initializer.name: numpy_helper.to_array(initializer) for initializer in model.graph.initializer}

def normalise_name(name: str) -> str:
    return name.lower().strip()

def weight_filter_scores(weights: np.ndarray) -> np.ndarray:
    """Calculate importance scores for filters based on L2 norm."""
    axes = tuple(range(1, weights.ndim))
    squared = np.square(weights, dtype=np.float32)
    return np.sqrt(np.sum(squared, axis=axes))

def cluster_scores(scores: np.ndarray, cluster_size: int) -> List[Tuple[np.ndarray, float]]:
    """Cluster scores and calculate average score for each cluster."""
    if cluster_size <= 0:
        raise ValueError("Cluster size must be a positive integer.")
    num_filters = scores.shape[0]
    clusters = []
    for i in range(0, num_filters, cluster_size):
        cluster_indices = np.arange(i, min(i + cluster_size, num_filters))
        cluster_score = float(np.mean(scores[cluster_indices]))
        clusters.append((cluster_indices, cluster_score))
    return clusters