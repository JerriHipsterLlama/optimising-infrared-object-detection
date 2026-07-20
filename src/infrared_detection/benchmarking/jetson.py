"""Jetson-only TensorRT benchmark interface."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


class JetsonToolsUnavailable(RuntimeError):
    """Raised when a Jetson benchmark is requested outside a configured Jetson."""


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise JetsonToolsUnavailable(
            f"Jetson benchmark requires '{name}', which is unavailable. Run this on the Jetson with TensorRT installed."
        )
    return path


def benchmark_tensorrt_engine(
    engine_path: str | Path,
    iterations: int = 100,
    warmup: int = 20,
    include_preprocess: bool = False,
    include_postprocess: bool = False,
) -> dict[str, Any]:
    """Benchmark a prebuilt engine using trtexec and normalize timing output."""

    del include_preprocess, include_postprocess
    engine = Path(engine_path)
    if not engine.exists():
        raise FileNotFoundError(engine)
    trtexec = _require_tool("trtexec")
    command = [
        trtexec,
        f"--loadEngine={engine}",
        f"--iterations={int(iterations)}",
        f"--warmUp={int(warmup)}",
        "--noDataTransfers",
        "--verbose",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    output = completed.stdout + "\n" + completed.stderr
    match = re.search(r"Latency: min = [^,]+, max = [^,]+, mean = [^,]+, median = ([0-9.]+)", output)
    if not match:
        raise RuntimeError("Could not parse TensorRT latency from trtexec output.")
    median_ms = float(match.group(1))
    return {
        "latency_p50_ms": median_ms,
        "latency_p95_ms": None,
        "fps": 1000.0 / median_ms if median_ms else None,
        "peak_memory_mb": None,
        "power_w": None,
        "energy_mj_per_inference": None,
        "engine_path": str(engine.resolve()),
        "iterations": int(iterations),
        "warmup": int(warmup),
    }

