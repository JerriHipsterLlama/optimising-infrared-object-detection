from __future__ import annotations

import pytest
import torch
from torch import nn

from infrared_detection.benchmarking.pytorch_cuda import benchmark_pytorch_cuda_forward


class FakeTimer:
    def __init__(self, values):
        self.values = iter(values)
        self.synchronizations = 0

    def synchronize(self):
        self.synchronizations += 1

    def measure(self, operation):
        operation()
        return next(self.values)


def test_forward_profiler_warms_up_and_reports_distribution():
    model = nn.Identity().eval()
    batch = torch.zeros(1, 3, 8, 8)
    timer = FakeTimer([1.0, 2.0, 4.0, 8.0])

    result = benchmark_pytorch_cuda_forward(model, batch, warmup=2, iterations=4, timer=timer)

    assert result["latency_mean_ms"] == pytest.approx(3.75)
    assert result["latency_p50_ms"] == pytest.approx(3.0)
    assert result["latency_p95_ms"] == pytest.approx(7.4)
    assert result["fps"] == pytest.approx(1000.0 / 3.75)
    assert result["warmup"] == 2
    assert result["iterations"] == 4
    assert result["input_shape"] == [1, 3, 8, 8]
    assert timer.synchronizations == 2


@pytest.mark.parametrize(
    "kwargs",
    [{"warmup": -1, "iterations": 1}, {"warmup": 0, "iterations": 0}, {"warmup": 0, "iterations": -1}],
)
def test_forward_profiler_rejects_invalid_counts(kwargs):
    with pytest.raises(ValueError):
        benchmark_pytorch_cuda_forward(nn.Identity(), torch.zeros(1), timer=FakeTimer([]), **kwargs)


def test_forward_profiler_requires_cuda_for_production_timer():
    with pytest.raises(ValueError, match="CUDA"):
        benchmark_pytorch_cuda_forward(nn.Identity(), torch.zeros(1), warmup=0, iterations=1)
