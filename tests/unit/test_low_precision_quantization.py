"""Tests for the low precision quantization utilities."""

import tempfile
import unittest

import torch

from infrared_detection.compression.quantization import (
    dequantize_state_dict,
    quantize_checkpoint,
    quantize_state_dict,
)


class TestLowPrecisionQuantization(unittest.TestCase):
    def setUp(self):
        self.state_dict = {
            "layer.weight": torch.tensor(
                [[-1.5, -0.5, 0.25, 1.75], [2.5, -3.0, 4.0, -4.5]],
                dtype=torch.float32,
            ),
            "layer.bias": torch.tensor([0.125, -0.375], dtype=torch.float64),
            "batch_norm.num_batches_tracked": torch.tensor(7, dtype=torch.int64),
        }

    def test_quantize_state_dict_uses_integer_storage(self):
        quantized_state_dict, metadata, stats = quantize_state_dict(self.state_dict, 8)

        self.assertEqual(quantized_state_dict["layer.weight"].dtype, torch.int8)
        self.assertEqual(quantized_state_dict["layer.bias"].dtype, torch.int8)
        self.assertEqual(metadata["layer.weight"]["storage_dtype"], "int8")
        self.assertIn("layer.weight", stats)
        self.assertEqual(quantized_state_dict["batch_norm.num_batches_tracked"].dtype, torch.int64)
        self.assertFalse(metadata["batch_norm.num_batches_tracked"]["is_quantized"])

    def test_quantize_and_dequantize_round_trip(self):
        quantized_state_dict, metadata, _ = quantize_state_dict(self.state_dict, 8)
        restored = dequantize_state_dict(quantized_state_dict, metadata)

        self.assertTrue(torch.allclose(restored["layer.weight"], self.state_dict["layer.weight"], atol=0.05, rtol=0.05))
        self.assertTrue(torch.allclose(restored["layer.bias"], self.state_dict["layer.bias"].to(torch.float64), atol=0.02, rtol=0.05))

    def test_checkpoint_quantization_returns_expected_payload(self):
        payload = quantize_checkpoint(self.state_dict, 2)

        self.assertEqual(payload["bit_width"], 2)
        self.assertIn("quantized_state_dict", payload)
        self.assertIn("metadata", payload)
        self.assertIn("tensor_stats", payload)
        self.assertEqual(payload["quantized_state_dict"]["layer.weight"].dtype, torch.int8)


if __name__ == "__main__":
    unittest.main()
