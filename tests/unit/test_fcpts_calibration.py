"""Tests for FCPTS calibration and differentiable pruning utilities."""

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from infrared_detection.compression.pruning.fcpts import (
    CalibrationRunner,
    DifferentiablePruningMaskFn,
    calibrate_model,
    finalize_and_export,
)


class _ToyDataset(Dataset):
    def __init__(self, size: int = 8):
        self.inputs = [torch.randn(4) for _ in range(size)]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]


class _ToyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 6)
        self.head = nn.Linear(6, 3)

    def forward(self, x):
        feat = torch.tanh(self.backbone(x))
        logits = self.head(feat)
        return {"feat": feat, "logits": logits}


def _toy_adapter(model, batch, targets=None, params_override=None):
    if params_override is not None:
        from torch.func import functional_call

        output = functional_call(model, params_override, (batch,))
    else:
        output = model(batch)
    return [output["feat"], output["logits"]]


class TestDifferentiablePruningMaskFn(unittest.TestCase):
    def test_forward_matches_mask_equation(self):
        weights = torch.tensor([-2.0, -0.2, 0.0, 0.3, 2.5], dtype=torch.float32)
        threshold = torch.tensor(0.25, dtype=torch.float32)

        mask = DifferentiablePruningMaskFn.apply(weights, threshold, 0.2)
        expected = 0.5 * torch.sign(torch.abs(weights) - threshold) + 0.5

        self.assertTrue(torch.equal(mask, expected))

    def test_threshold_grad_from_sparsity_path_is_finite(self):
        weights = torch.tensor([-0.9, -0.2, 0.1, 0.7, 1.1], dtype=torch.float32, requires_grad=True)
        threshold = torch.tensor(0.3, dtype=torch.float32, requires_grad=True)

        mask = DifferentiablePruningMaskFn.apply(weights, threshold, 0.15)
        sparsity = 1.0 - mask.mean()
        sparsity.backward()

        self.assertIsNotNone(threshold.grad)
        self.assertTrue(torch.isfinite(threshold.grad))


class TestCalibrationRunner(unittest.TestCase):
    def test_runner_computes_total_reconstruction_and_control_losses(self):
        dense_model = _ToyDetector()
        sparse_model = _ToyDetector()
        runner = CalibrationRunner(
            dense_model=dense_model,
            sparse_model=sparse_model,
            output_adapter=_toy_adapter,
            target_sparsity=0.35,
            lambda_control=1.0,
            bandwidth=0.2,
        )
        batch = torch.randn(2, 4)

        losses = runner.compute_losses(batch)

        self.assertIn("loss", losses)
        self.assertIn("l_rec", losses)
        self.assertIn("l_c", losses)
        self.assertIn("global_sparsity", losses)
        self.assertGreaterEqual(float(losses["global_sparsity"].detach().item()), 0.0)
        self.assertLessEqual(float(losses["global_sparsity"].detach().item()), 1.0)

    def test_calibration_loop_writes_csv_metrics(self):
        dense_model = _ToyDetector()
        sparse_model = _ToyDetector()
        runner = CalibrationRunner(
            dense_model=dense_model,
            sparse_model=sparse_model,
            output_adapter=_toy_adapter,
            target_sparsity=0.4,
            lambda_control=0.7,
            bandwidth=0.25,
        )
        loader = DataLoader(_ToyDataset(size=6), batch_size=2)

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "calibration_metrics.csv"
            calibrate_model(
                runner=runner,
                dataloader=loader,
                epochs=1,
                lr=1e-3,
                metrics_csv_path=csv_path,
                device=torch.device("cpu"),
            )
            self.assertTrue(csv_path.exists())
            lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertGreaterEqual(len(lines), 2)

    def test_finalize_and_export_applies_permanent_mask(self):
        sparse_model = _ToyDetector()
        thresholds = {}
        for name, param in sparse_model.named_parameters():
            if param.ndim > 1:
                thresholds[name] = torch.tensor(1e6)

        finalized = finalize_and_export(
            sparse_model=sparse_model,
            threshold_registry=thresholds,
            bandwidth=0.2,
        )

        for name, param in finalized.named_parameters():
            if param.ndim > 1:
                self.assertEqual(torch.count_nonzero(param).item(), 0)


if __name__ == "__main__":
    unittest.main()
