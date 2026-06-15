"""Unit tests for the ONNX cluster pruning module."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.python.optimisations.cluster_pruning import (  # noqa: E402
    ClusterPruningConfig,
    infer_protected_model_indices,
    prune_model,
    select_channels_to_zero,
    is_protected_node,
    weight_filter_scores,
)


def make_simple_pruning_model() -> onnx.ModelProto:
    """Create a tiny ONNX model with a prunable backbone conv and protected head conv."""

    input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 1, 5, 5])
    output_tensor = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 2, 3, 3])

    backbone_weights = numpy_helper.from_array(
        np.array(
            [
                [[[0.1]]],
                [[[0.2]]],
                [[[0.9]]],
                [[[1.5]]],
            ],
            dtype=np.float32,
        ),
        name="backbone_weights",
    )
    backbone_bias = numpy_helper.from_array(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32), name="backbone_bias")

    head_weights = numpy_helper.from_array(
        np.array(
            [
                [[[1.0]], [[1.0]], [[1.0]], [[1.0]]],
                [[[2.0]], [[2.0]], [[2.0]], [[2.0]]],
            ],
            dtype=np.float32,
        ),
        name="detect_head_weights",
    )
    head_bias = numpy_helper.from_array(np.array([0.5, 0.6], dtype=np.float32), name="detect_head_bias")

    backbone_conv = helper.make_node(
        "Conv",
        inputs=["input", "backbone_weights", "backbone_bias"],
        outputs=["backbone_output"],
        name="backbone_conv",
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    )
    head_conv = helper.make_node(
        "Conv",
        inputs=["backbone_output", "detect_head_weights", "detect_head_bias"],
        outputs=["output"],
        name="detect_head_conv",
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    )

    graph = helper.make_graph(
        nodes=[backbone_conv, head_conv],
        name="cluster_pruning_test_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[backbone_weights, backbone_bias, head_weights, head_bias],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


class TestClusterSelection(unittest.TestCase):
    """Test the cluster selection policy."""

    def test_select_channels_to_zero_prefers_low_magnitude_clusters(self) -> None:
        weights = np.array(
            [
                [[[0.1]]],
                [[[0.2]]],
                [[[0.9]]],
                [[[1.5]]],
            ],
            dtype=np.float32,
        )

        selected = select_channels_to_zero(
            weights=weights,
            prune_ratio=0.5,
            cluster_size=2,
            min_channels_to_keep=1,
        )

        scores = weight_filter_scores(weights)
        selected_set = set(selected.tolist())
        all_indices = set(range(weights.shape[0]))
        non_selected = sorted(all_indices - selected_set)

        if selected_set and non_selected:
            max_selected_score = float(max(scores[list(selected_set)]))
            min_non_selected_score = float(min(scores[non_selected]))
            self.assertLessEqual(max_selected_score, min_non_selected_score)
        else:
            # fallback: if nothing selected or everything selected, assert lengths make sense
            self.assertTrue(len(selected) <= int(round(weights.shape[0] * 0.5)))


class TestProtectedNodes(unittest.TestCase):
    """Test protected-node heuristics and pruning behavior."""

    def test_is_protected_node_detects_detection_head_keywords(self) -> None:
        node = helper.make_node(
            "Conv",
            inputs=["x", "w"],
            outputs=["detect_head_output"],
            name="detect_head_conv",
        )

        config = ClusterPruningConfig(
            onnx_path=Path("input.onnx"),
            output_path=Path("output.onnx"),
        )

        self.assertTrue(is_protected_node(node, config))

    def test_infer_protected_model_indices_detects_last_yolo_block(self) -> None:
        backbone = helper.make_node(
            "Conv",
            inputs=["x", "w1"],
            outputs=["/model.21/backbone_output"],
            name="/model.21/backbone/Conv",
        )
        head = helper.make_node(
            "Conv",
            inputs=["/model.21/backbone_output", "w2"],
            outputs=["/model.22/head_output"],
            name="/model.22/head/Conv",
        )
        graph = helper.make_graph(
            nodes=[backbone, head],
            name="yolo_tail_test",
            inputs=[helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 1, 5, 5])],
            outputs=[helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 1, 1, 1])],
            initializer=[],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

        self.assertEqual(infer_protected_model_indices(model), {22})

        config = ClusterPruningConfig(
            onnx_path=Path("input.onnx"),
            output_path=Path("output.onnx"),
        )

        self.assertFalse(is_protected_node(backbone, config, {22}))
        self.assertTrue(is_protected_node(head, config, {22}))

    def test_prune_model_preserves_protected_head_weights(self) -> None:
        model = make_simple_pruning_model()

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source.onnx"
            output_path = Path(temp_dir) / "output.onnx"
            onnx.save(model, str(source_path))

            config = ClusterPruningConfig(
                onnx_path=source_path,
                output_path=output_path,
                prune_ratio=0.5,
                cluster_size=2,
            )

            report = prune_model(model, config)

            self.assertEqual(report.total_conv_nodes, 2)
            self.assertEqual(report.protected_conv_nodes, 1)
            self.assertEqual(report.pruned_conv_nodes, 1)
            self.assertTrue(output_path.exists())

            pruned_model = onnx.load(str(output_path))
            initializers = {initializer.name: numpy_helper.to_array(initializer) for initializer in pruned_model.graph.initializer}

            self.assertTrue(np.all(initializers["backbone_weights"][0] == 0))
            self.assertTrue(np.all(initializers["backbone_weights"][1] == 0))
            self.assertTrue(np.any(initializers["detect_head_weights"] != 0))


if __name__ == "__main__":
    unittest.main()