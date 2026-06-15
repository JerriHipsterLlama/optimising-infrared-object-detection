"""ONNX cluster pruning utilities for object detection models.

This module provides a conservative first implementation of cluster pruning
for ONNX object detection models. It focuses on backbone/body convolution
layers and keeps detection/classification heads intact by default.

The current implementation is intentionally safe:
- it loads an ONNX model,
- identifies prunable Conv nodes outside protected regions,
- scores output filter clusters by weight magnitude,
- zeros the selected filters in the weight tensor,
- and writes a valid ONNX model back to disk.

This preserves graph topology, which makes it a good starting point before
introducing more aggressive structural channel removal.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import onnx
from onnx import numpy_helper
from sklearn.cluster import KMeans
import subprocess
import time
import os
import onnxruntime as ort


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


MODEL_BLOCK_PATTERN = re.compile(r"/model\.(\d+)/")


@dataclass(slots=True)
class ClusterPruningConfig:
	"""Configuration for a pruning run."""

	onnx_path: Path
	output_path: Path
	prune_ratio: float = 0.2
	cluster_size: int = 4
	importance_method: str = "centroid"  # one of: 'centroid', 'weights', 'validation'
	protected_node_names: Set[str] = field(default_factory=set)
	protected_output_names: Set[str] = field(default_factory=set)
	protected_op_types: Set[str] = field(default_factory=lambda: set(DEFAULT_PROTECTED_OP_TYPES))
	protected_keywords: Tuple[str, ...] = DEFAULT_PROTECTED_KEYWORDS
	min_channels_to_keep: int = 1
	dry_run: bool = False
	structural: bool = False
	iterations: int = 1
	prune_decay: float = 1.0
	finetune_command: str | None = None


@dataclass(slots=True)
class PruningReport:
	"""Summary of a pruning pass."""

	source_model: Path
	output_model: Path
	total_conv_nodes: int
	protected_conv_nodes: int
	pruned_conv_nodes: int
	channels_zeroed: int
	modified_nodes: List[str]
	protected_nodes: List[str]


def load_model(model_path: Path) -> onnx.ModelProto:
	"""Load an ONNX model from disk."""

	if not model_path.exists():
		raise FileNotFoundError(f"ONNX model not found: {model_path}")

	return onnx.load(str(model_path))


def save_model(model: onnx.ModelProto, output_path: Path) -> None:
	"""Save an ONNX model to disk."""

	output_path.parent.mkdir(parents=True, exist_ok=True)
	onnx.save(model, str(output_path))


def build_initializer_map(model: onnx.ModelProto) -> Dict[str, np.ndarray]:
	"""Map initializer names to NumPy arrays."""

	return {
		initializer.name: numpy_helper.to_array(initializer)
		for initializer in model.graph.initializer
	}


def normalize_name(name: str) -> str:
	"""Normalize names for matching."""

	return name.lower().strip()


def has_protected_keyword(name: str, keywords: Sequence[str]) -> bool:
	"""Return True when a name contains a protected keyword."""

	lowered = normalize_name(name)
	return any(keyword in lowered for keyword in keywords)


def infer_protected_model_indices(model: onnx.ModelProto) -> Set[int]:
	"""Infer the last top-level model block index used by YOLO-style exports."""

	indices: Set[int] = set()
	for node in model.graph.node:
		for text in (node.name, *node.output):
			if not text:
				continue
			for match in MODEL_BLOCK_PATTERN.finditer(text):
				indices.add(int(match.group(1)))

	if not indices:
		return set()

	return {max(indices)}


def node_model_indices(node: onnx.NodeProto) -> Set[int]:
	"""Return any top-level model block indices referenced by a node."""

	indices: Set[int] = set()
	for text in (node.name, *node.output):
		if not text:
			continue
		for match in MODEL_BLOCK_PATTERN.finditer(text):
			indices.add(int(match.group(1)))
	return indices


def is_protected_node(
	node: onnx.NodeProto,
	config: ClusterPruningConfig,
	protected_model_indices: Set[int] | None = None,
) -> bool:
	"""Determine whether a node must not be pruned."""

	if node.name and node.name in config.protected_node_names:
		return True

	if node.op_type in config.protected_op_types:
		return True

	if node.output and any(output in config.protected_output_names for output in node.output):
		return True

	if node.name and has_protected_keyword(node.name, config.protected_keywords):
		return True

	if protected_model_indices and node_model_indices(node) & protected_model_indices:
		return True

	return any(has_protected_keyword(output, config.protected_keywords) for output in node.output)


def is_prunable_conv(node: onnx.NodeProto) -> bool:
	"""Return True for Conv nodes that can be safely considered for pruning."""

	return node.op_type == "Conv"


def weight_filter_scores(weights: np.ndarray) -> np.ndarray:
	"""Score each output filter by L2 magnitude."""

	if weights.ndim < 2:
		raise ValueError(f"Conv weight tensor must have at least 2 dimensions, got {weights.shape}")

	axes = tuple(range(1, weights.ndim))
	squared = np.square(weights, dtype=np.float32)
	return np.sqrt(np.sum(squared, axis=axes))


def cluster_scores(scores: np.ndarray, cluster_size: int) -> List[Tuple[np.ndarray, float]]:
	"""Group filter scores into contiguous clusters and compute cluster scores."""

	if cluster_size <= 0:
		raise ValueError("cluster_size must be a positive integer")

	clusters: List[Tuple[np.ndarray, float]] = []
	for start in range(0, len(scores), cluster_size):
		indices = np.arange(start, min(start + cluster_size, len(scores)))
		cluster_score = float(np.mean(scores[indices]))
		clusters.append((indices, cluster_score))
	return clusters


def select_channels_to_zero(
	weights: np.ndarray,
	prune_ratio: float,
	cluster_size: int,
	min_channels_to_keep: int,
	importance_method: str = "centroid",
) -> np.ndarray:
	"""Pick output channels to zero using k-means over flattened filter vectors.

	This groups similar filters with KMeans, computes cluster scores as the
	mean L2 score of cluster members, then selects lowest-scoring clusters
	until the target number of channels to prune is reached.
	"""

	if prune_ratio <= 0:
		return np.array([], dtype=np.int64)

	if weights.ndim != 4:
		# The first implementation only handles standard Conv kernels.
		return np.array([], dtype=np.int64)

	out_channels = weights.shape[0]
	if out_channels <= min_channels_to_keep:
		return np.array([], dtype=np.int64)

	total_prunable = max(0, out_channels - min_channels_to_keep)
	target_channels = int(round(out_channels * prune_ratio))
	target_channels = min(target_channels, total_prunable)
	if target_channels <= 0:
		return np.array([], dtype=np.int64)

	# Flatten each filter to a vector (out_channels, -1)
	flattened = weights.reshape(out_channels, -1)

	# Number of k-means clusters (at least 1)
	n_clusters = max(1, out_channels // cluster_size)

	# Run KMeans to group similar filters
	kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
	labels = kmeans.fit_predict(flattened)

	# Score each filter by L2 magnitude
	filter_scores = weight_filter_scores(weights)

	# Compute cluster scores
	clusters: List[Tuple[np.ndarray, float]] = []
	if importance_method == "centroid":
		# use centroid vector norms as cluster importance
		centers = kmeans.cluster_centers_
		center_norms = np.linalg.norm(centers, axis=1)
		for c in range(n_clusters):
			members = np.where(labels == c)[0]
			if members.size == 0:
				continue
			cluster_score = float(center_norms[c])
			clusters.append((members, cluster_score))
	else:
		# fallback to mean filter L2 in cluster
		for c in range(n_clusters):
			members = np.where(labels == c)[0]
			if members.size == 0:
				continue
			cluster_score = float(np.mean(filter_scores[members]))
			clusters.append((members, cluster_score))

	# Sort clusters from least important (low score) to most important
	clusters_sorted = sorted(clusters, key=lambda item: item[1])

	selected: List[int] = []
	for members, _cluster_score in clusters_sorted:
		if len(selected) + len(members) > target_channels and selected:
			break
		selected.extend(int(int_idx) for int_idx in members.tolist())
		if len(selected) >= target_channels:
			break

	if not selected:
		return np.array([], dtype=np.int64)

	selected_array = np.array(sorted(set(selected)), dtype=np.int64)
	if selected_array.size > total_prunable:
		selected_array = selected_array[:total_prunable]
	return selected_array


def zero_conv_filters(weights: np.ndarray, channels_to_zero: np.ndarray) -> np.ndarray:
	"""Zero the selected output channels in a Conv kernel."""

	if channels_to_zero.size == 0:
		return weights

	updated = weights.copy()
	updated[channels_to_zero] = 0
	return updated


def update_initializer_tensor(
	model: onnx.ModelProto,
	initializer_map: Dict[str, np.ndarray],
	tensor_name: str,
	updated_array: np.ndarray,
) -> None:
	"""Replace an initializer tensor in the model graph."""

	initializer_map[tensor_name] = updated_array
	for index, initializer in enumerate(model.graph.initializer):
		if initializer.name == tensor_name:
			model.graph.initializer[index].CopyFrom(numpy_helper.from_array(updated_array, tensor_name))
			return

	raise KeyError(f"Initializer not found in graph: {tensor_name}")


def prune_model(model: onnx.ModelProto, config: ClusterPruningConfig) -> PruningReport:
	"""Apply conservative cluster pruning to an ONNX model."""

	initializer_map = build_initializer_map(model)
	protected_model_indices = infer_protected_model_indices(model)
	modified_nodes: List[str] = []
	protected_nodes: List[str] = []

	total_conv_nodes = 0
	protected_conv_nodes = 0
	pruned_conv_nodes = 0
	channels_zeroed = 0

	for node in model.graph.node:
		if not is_prunable_conv(node):
			continue

		total_conv_nodes += 1

		if is_protected_node(node, config, protected_model_indices):
			protected_conv_nodes += 1
			protected_nodes.append(node.name or node.output[0] if node.output else node.op_type)
			continue

		if len(node.input) < 2:
			continue

		weight_name = node.input[1]
		weights = initializer_map.get(weight_name)
		if weights is None:
			continue

		channels_to_zero = select_channels_to_zero(
			weights=weights,
			prune_ratio=config.prune_ratio,
			cluster_size=config.cluster_size,
			min_channels_to_keep=config.min_channels_to_keep,
			importance_method=getattr(config, "importance_method", "centroid"),
		)

		if channels_to_zero.size == 0:
			continue

		updated_weights = zero_conv_filters(weights, channels_to_zero)
		update_initializer_tensor(model, initializer_map, weight_name, updated_weights)

		if len(node.input) >= 3:
			bias_name = node.input[2]
			bias = initializer_map.get(bias_name)
			if bias is not None and bias.ndim == 1 and bias.shape[0] == weights.shape[0]:
				updated_bias = bias.copy()
				updated_bias[channels_to_zero] = 0
				update_initializer_tensor(model, initializer_map, bias_name, updated_bias)

		pruned_conv_nodes += 1
		channels_zeroed += int(channels_to_zero.size)
		modified_nodes.append(node.name or node.output[0] if node.output else weight_name)

	if not config.dry_run:
		model = onnx.shape_inference.infer_shapes(model)
		save_model(model, config.output_path)

	return PruningReport(
		source_model=config.onnx_path,
		output_model=config.output_path,
		total_conv_nodes=total_conv_nodes,
		protected_conv_nodes=protected_conv_nodes,
		pruned_conv_nodes=pruned_conv_nodes,
		channels_zeroed=channels_zeroed,
		modified_nodes=modified_nodes,
		protected_nodes=protected_nodes,
	)


def find_consumers(model: onnx.ModelProto, tensor_name: str) -> List[onnx.NodeProto]:
	"""Return list of nodes that consume a given tensor."""

	consumers: List[onnx.NodeProto] = []
	for node in model.graph.node:
		if tensor_name in node.input:
			consumers.append(node)
	return consumers


def structural_prune_model(model: onnx.ModelProto, config: ClusterPruningConfig) -> PruningReport:
	"""Perform structural channel removal when safe (Conv -> (BN) -> Conv chains).

	This function attempts to remove pruned output channels from a Conv's
	weight initializer and propagate the removal to the following BatchNorm
	and Conv initializers where possible. It will skip nodes with complex
	consumers (concat, split, multiple branches).
	"""

	initializer_map = build_initializer_map(model)
	protected_model_indices = infer_protected_model_indices(model)
	modified_nodes: List[str] = []
	protected_nodes: List[str] = []

	total_conv_nodes = 0
	protected_conv_nodes = 0
	pruned_conv_nodes = 0
	channels_removed = 0

	# We'll operate on a copy to avoid iterator invalidation
	for node in list(model.graph.node):
		if not is_prunable_conv(node):
			continue
		total_conv_nodes += 1

		if is_protected_node(node, config, protected_model_indices):
			protected_conv_nodes += 1
			protected_nodes.append(node.name or (node.output[0] if node.output else node.op_type))
			continue

		if len(node.input) < 2:
			continue

		weight_name = node.input[1]
		weights = initializer_map.get(weight_name)
		if weights is None or weights.ndim != 4:
			continue

		# Determine channels to remove using same selection logic
		channels_to_remove = select_channels_to_zero(
			weights=weights,
			prune_ratio=config.prune_ratio,
			cluster_size=config.cluster_size,
			min_channels_to_keep=config.min_channels_to_keep,
			importance_method=getattr(config, "importance_method", "centroid"),
		)

		if channels_to_remove.size == 0:
			continue

		# Find consumers of this conv's output
		output_tensor = node.output[0] if node.output else None
		if output_tensor is None:
			continue

		# Helper to find all next Convs in a potentially branching/activations path
		def get_structural_update_targets(current_tensor: str, visited: Set[str]) -> Tuple[bool, List[onnx.NodeProto], List[onnx.NodeProto]]:
			"""Find all downstream nodes (Convs, BNs) that need updating if current_tensor changes shape."""
			if current_tensor in visited:
				return True, [], [] # Already handled
			visited.add(current_tensor)
			
			curr_consumers = find_consumers(model, current_tensor)
			bns: List[onnx.NodeProto] = []
			convs: List[onnx.NodeProto] = []
			
			for c in curr_consumers:
				if c.op_type == "Conv":
					# Only if current_tensor is the data input (index 0)
					if c.input[0] == current_tensor:
						convs.append(c)
					else:
						# If it's a weight/bias, this is weird.
						return False, [], []
				elif c.op_type == "BatchNormalization":
					bns.append(c)
					success, sub_convs, sub_bns = get_structural_update_targets(c.output[0], visited)
					if not success: return False, [], []
					convs.extend(sub_convs)
					bns.extend(sub_bns)
				elif c.op_type in ("Relu", "LeakyRelu", "Sigmoid", "HardSigmoid", "SiLU", "Swish", "Mul"):
					# Note: Mul must be element-wise. We assume simple Mul here.
					success, sub_convs, sub_bns = get_structural_update_targets(c.output[0], visited)
					if not success: return False, [], []
					convs.extend(sub_convs)
					bns.extend(sub_bns)
				else:
					# Complex consumer (Concat, Resize, Split, etc.) - cannot safely prune structurally
					return False, [], []
			
			return True, convs, bns

		success, next_conv_nodes, bn_nodes = get_structural_update_targets(output_tensor, set())

		# Safety: do not structurally modify if no next Conv was found or if any are protected
		if not success or not next_conv_nodes:
			continue
		
		# check all next convs
		if any(is_protected_node(nc, config, protected_model_indices) for nc in next_conv_nodes):
			continue

		# Remove channels from this conv's weight initializer
		new_weights = np.delete(weights, channels_to_remove, axis=0)
		update_initializer_tensor(model, initializer_map, weight_name, new_weights)
		# Update next Conv's input channels for ALL found next convs
		for next_conv_node in next_conv_nodes:
			if len(next_conv_node.input) >= 2:
				next_weight_name = next_conv_node.input[1]
				next_weights = initializer_map.get(next_weight_name)
				if next_weights is not None and next_weights.ndim == 4:
					new_next_weights = np.delete(next_weights, channels_to_remove, axis=1)
					update_initializer_tensor(model, initializer_map, next_weight_name, new_next_weights)

		pruned_conv_nodes += 1
		channels_removed += int(channels_to_remove.size)
		modified_nodes.append(node.name or (node.output[0] if node.output else weight_name))

	# finalize
	model = onnx.shape_inference.infer_shapes(model)
	if not config.dry_run:
		save_model(model, config.output_path)

	return PruningReport(
		source_model=config.onnx_path,
		output_model=config.output_path,
		total_conv_nodes=total_conv_nodes,
		protected_conv_nodes=protected_conv_nodes,
		pruned_conv_nodes=pruned_conv_nodes,
		channels_zeroed=channels_removed,
		modified_nodes=modified_nodes,
		protected_nodes=protected_nodes,
	)


def iterative_prune_and_finetune(
	onnx_path: Path,
	output_path: Path,
	config: ClusterPruningConfig,
) -> None:
	"""Run iterative prune -> optional finetune cycles according to config."""

	working_path = Path(onnx_path)
	for i in range(config.iterations):
		print(f"Iteration {i+1}/{config.iterations}: prune_ratio={config.prune_ratio}")
		model = load_model(working_path)
		if config.structural:
			report = structural_prune_model(model, config)
		else:
			report = prune_model(model, config)

		print(f"  Pruned nodes: {report.pruned_conv_nodes}, channels affected: {report.channels_zeroed}")

		# update working path to the last output
		working_path = report.output_model

		# optional finetune step
		if config.finetune_command:
			print(f"  Running finetune command: {config.finetune_command}")
			completed = subprocess.run(config.finetune_command, shell=True)
			if completed.returncode != 0:
				raise RuntimeError(f"Finetune command failed with code {completed.returncode}")

		# decay prune ratio if requested
		config.prune_ratio *= config.prune_decay


def benchmark_onnx(model_path: Path, warmup: int = 5, runs: int = 50) -> Dict[str, float]:
	"""Measure simple ONNX inference latency and model size.

	Returns a dict with `size_bytes`, `median_ms`, `mean_ms`.
	"""

	sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
	input_meta = sess.get_inputs()[0]
	input_shape = [dim if isinstance(dim, int) else 1 for dim in input_meta.shape]
	dummy = np.random.randn(*input_shape).astype(np.float32)

	# warmup
	for _ in range(warmup):
		sess.run(None, {input_meta.name: dummy})

	times = []
	for _ in range(runs):
		t0 = time.time()
		sess.run(None, {input_meta.name: dummy})
		t1 = time.time()
		times.append((t1 - t0) * 1000.0)

	size_bytes = os.path.getsize(model_path)
	times_arr = np.array(times)
	return {
		"size_bytes": float(size_bytes),
		"median_ms": float(np.median(times_arr)),
		"mean_ms": float(np.mean(times_arr)),
	}


def parse_comma_separated(values: Iterable[str] | None) -> Set[str]:
	"""Normalize CLI list values into a set."""

	if not values:
		return set()

	items: Set[str] = set()
	for value in values:
		for part in value.split(","):
			cleaned = part.strip()
			if cleaned:
				items.add(cleaned)
	return items


def build_config_from_args(args: argparse.Namespace) -> ClusterPruningConfig:
	"""Build a pruning config from CLI arguments."""

	protected_keywords = tuple(
		keyword.strip().lower()
		for keyword in args.protected_keywords
		if keyword.strip()
	) if args.protected_keywords else DEFAULT_PROTECTED_KEYWORDS

	return ClusterPruningConfig(
		onnx_path=Path(args.onnx_path),
		output_path=Path(args.output_path),
		prune_ratio=args.prune_ratio,
		cluster_size=args.cluster_size,
		protected_node_names=parse_comma_separated(args.protected_node_names),
		protected_output_names=parse_comma_separated(args.protected_output_names),
		protected_op_types=parse_comma_separated(args.protected_op_types) or set(DEFAULT_PROTECTED_OP_TYPES),
		protected_keywords=protected_keywords,
		min_channels_to_keep=args.min_channels_to_keep,
		dry_run=args.dry_run,
		structural=getattr(args, "structural", False),
	)


def build_parser() -> argparse.ArgumentParser:
	"""Create the command line parser."""

	parser = argparse.ArgumentParser(
		description="Apply conservative cluster pruning to an ONNX object detection model.",
	)
	parser.add_argument("onnx_path", help="Path to the source ONNX model")
	parser.add_argument(
		"output_path",
		help="Where to write the pruned ONNX model",
	)
	parser.add_argument(
		"--prune-ratio",
		type=float,
		default=0.2,
		help="Fraction of filters to zero in each eligible Conv layer",
	)
	parser.add_argument(
		"--cluster-size",
		type=int,
		default=4,
		help="Number of consecutive filters per pruning cluster",
	)
	parser.add_argument(
		"--protected-node-names",
		nargs="*",
		default=[],
		help="Node names that must not be pruned",
	)
	parser.add_argument(
		"--protected-output-names",
		nargs="*",
		default=[],
		help="Node output tensor names that must not be pruned",
	)
	parser.add_argument(
		"--protected-op-types",
		nargs="*",
		default=list(DEFAULT_PROTECTED_OP_TYPES),
		help="ONNX op types that must not be pruned",
	)
	parser.add_argument(
		"--protected-keywords",
		nargs="*",
		default=list(DEFAULT_PROTECTED_KEYWORDS),
		help="Name keywords that mark a node as protected",
	)
	parser.add_argument(
		"--min-channels-to-keep",
		type=int,
		default=1,
		help="Minimum number of output channels to preserve per Conv layer",
	)
	parser.add_argument(
		"--structural",
		action="store_true",
		help="Physically remove channels from weights instead of zeroing them (requires standard Conv/BN/Conv chains)",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Calculate the pruning plan without writing an output model",
	)
	return parser


def main() -> None:
	"""CLI entry point."""

	parser = build_parser()
	args = parser.parse_args()
	config = build_config_from_args(args)

	model = load_model(config.onnx_path)
	if config.structural:
		report = structural_prune_model(model, config)
	else:
		report = prune_model(model, config)

	print("Cluster pruning completed")
	print(f"  Source model: {report.source_model}")
	print(f"  Output model: {report.output_model}")
	print(f"  Conv nodes considered: {report.total_conv_nodes}")
	print(f"  Protected Conv nodes: {report.protected_conv_nodes}")
	print(f"  Pruned Conv nodes: {report.pruned_conv_nodes}")
	print(f"  Channels zeroed: {report.channels_zeroed}")
	if report.modified_nodes:
		print("  Modified nodes:")
		for node_name in report.modified_nodes:
			print(f"    - {node_name}")
	if report.protected_nodes:
		print("  Protected nodes:")
		for node_name in report.protected_nodes:
			print(f"    - {node_name}")


if __name__ == "__main__":
	main()

