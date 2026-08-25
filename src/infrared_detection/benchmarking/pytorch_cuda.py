"""Direct PyTorch CUDA forward-pass benchmarking."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any, Protocol

import torch


class ForwardTimer(Protocol):
    def synchronize(self) -> None: ...

    def measure(self, operation: Callable[[], Any]) -> float: ...


class CudaEventTimer:
    """Measure an operation with CUDA events, returning milliseconds."""

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def measure(self, operation: Callable[[], Any]) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end))


class CpuWallClockTimer:
    """Measure a CPU operation with a monotonic wall clock."""

    def synchronize(self) -> None:
        return None

    def measure(self, operation: Callable[[], Any]) -> float:
        start = time.perf_counter()
        operation()
        return (time.perf_counter() - start) * 1000.0


def benchmark_pytorch_cuda_forward(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
    timer: ForwardTimer | None = None,
) -> dict[str, Any]:
    """Benchmark direct model forward latency in milliseconds."""
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if timer is None:
        timer = CudaEventTimer() if example_input.is_cuda else CpuWallClockTimer()

    with torch.inference_mode():
        for _ in range(warmup):
            model(example_input)
        timer.synchronize()
        samples = [timer.measure(lambda: model(example_input)) for _ in range(iterations)]
        timer.synchronize()

    mean_ms = statistics.mean(samples)
    return {
        "latency_mean_ms": mean_ms,
        "latency_p50_ms": statistics.median(samples),
        "latency_p95_ms": float(torch.quantile(torch.tensor(samples), 0.95)),
        "fps": 1000.0 / mean_ms if mean_ms else None,
        "warmup": warmup,
        "iterations": iterations,
        "input_shape": list(example_input.shape),
    }
