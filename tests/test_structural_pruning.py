"""Integration tests for structural pruning behavior on YOLO-like ONNX graphs."""

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
    structural_prune_model,
)


def make_yolo_like_chain(protect_next_conv: bool = False) -> onnx.ModelProto:
    """Create a small ONNX model: Conv -> BN -> Conv chain.

    If `protect_next_conv` is True, name the second Conv with a protected
    keyword (e.g., 'detect') so structural pruning should skip it.
    """

    input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 1, 16, 16])
    output_tensor = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 8, 14, 14])

    # backbone conv: 8 output channels, 1 input channel, kernel 3x3
    backbone_w = np.random.randn(8, 1, 3, 3).astype(np.float32)
    backbone_b = np.random.randn(8).astype(np.float32)

    # next conv: 4 output channels, input channels = 8
    next_w = np.random.randn(4, 8, 3, 3).astype(np.float32)
    next_b = np.random.randn(4).astype(np.float32)

    backbone_weights = numpy_helper.from_array(backbone_w, name="backbone_w")
    backbone_bias = numpy_helper.from_array(backbone_b, name="backbone_b")
    next_weights = numpy_helper.from_array(next_w, name="next_w")
    next_bias = numpy_helper.from_array(next_b, name="next_b")

    backbone_conv = helper.make_node(
        "Conv",
        inputs=["input", "backbone_w", "backbone_b"],
        outputs=["backbone_out"],
        name="backbone_conv",
    )

    bn_scale = numpy_helper.from_array(np.ones(8, dtype=np.float32), name="bn_scale")
    bn_bias = numpy_helper.from_array(np.zeros(8, dtype=np.float32), name="bn_bias")
    bn_mean = numpy_helper.from_array(np.zeros(8, dtype=np.float32), name="bn_mean")
    bn_var = numpy_helper.from_array(np.ones(8, dtype=np.float32), name="bn_var")

    bn_node = helper.make_node(
        "BatchNormalization",
        inputs=["backbone_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        outputs=["bn_out"],
        name="backbone_bn",
    )

    next_name = "detect_head_conv" if protect_next_conv else "intermediate_conv"
    next_conv = helper.make_node(
        "Conv",
        inputs=["bn_out", "next_w", "next_b"],
        outputs=["output"],
        name=next_name,
    )

    graph = helper.make_graph(
        nodes=[backbone_conv, bn_node, next_conv],
        name="yolo_like_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[backbone_weights, backbone_bias, bn_scale, bn_bias, bn_mean, bn_var, next_weights, next_bias],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


class TestStructuralPruningIntegration(unittest.TestCase):
    def test_structural_prune_updates_next_conv_when_safe(self):
        model = make_yolo_like_chain(protect_next_conv=False)

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.onnx"
            out = Path(td) / "out.onnx"
            onnx.save(model, str(src))

            config = ClusterPruningConfig(onnx_path=src, output_path=out, prune_ratio=0.25, cluster_size=2, structural=True)
            report = structural_prune_model(model, config)

            self.assertTrue(report.pruned_conv_nodes >= 1)
            pruned = onnx.load(str(out))
            init = {i.name: numpy_helper.to_array(i) for i in pruned.graph.initializer}

            # next conv input channels should be reduced from 8
            self.assertTrue(init["next_w"].shape[1] < 8)

    def test_structural_prune_skips_when_next_conv_protected(self):
        model = make_yolo_like_chain(protect_next_conv=True)

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.onnx"
            out = Path(td) / "out.onnx"
            onnx.save(model, str(src))

            config = ClusterPruningConfig(onnx_path=src, output_path=out, prune_ratio=0.25, cluster_size=2, structural=True)
            report = structural_prune_model(model, config)

            # Should have skipped structural changes because next conv is protected
            pruned = onnx.load(str(out))
            init = {i.name: numpy_helper.to_array(i) for i in pruned.graph.initializer}

            # next conv input channels should remain 8
            self.assertEqual(init["next_w"].shape[1], 8)


if __name__ == "__main__":
    unittest.main()
