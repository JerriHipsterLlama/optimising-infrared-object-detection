"""Local RTX 3070 TensorRT build and benchmark adapters."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


class RtxToolsUnavailable(RuntimeError):
    """Raised when a local RTX benchmark cannot find TensorRT's trtexec tool."""


def _require_trtexec() -> str:
    trtexec = shutil.which("trtexec")
    if trtexec is None:
        raise RtxToolsUnavailable(
            "Local RTX TensorRT operation requires 'trtexec', which is unavailable. "
            "Install TensorRT and ensure trtexec is on PATH."
        )
    return trtexec


def _build_command(
    trtexec: str,
    onnx_path: Path,
    engine_path: Path,
    precision: str,
    calibration_dir: Path | None,
    workspace_mb: int,
) -> list[str]:
    command = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{int(workspace_mb)}",
    ]
    if precision == "fp16":
        command.append("--fp16")
    elif precision == "int8":
        command.extend(["--int8", f"--calib={calibration_dir}"])
    return command


def build_tensorrt_engine(
    onnx_path: Path,
    engine_path: Path,
    precision: str,
    calibration_dir: Path | None,
    workspace_mb: int,
) -> dict[str, Any]:
    """Build a local TensorRT engine using the installed ``trtexec`` binary."""

    if precision not in {"fp32", "fp16", "int8"}:
        raise ValueError("precision must be one of: fp32, fp16, int8")
    if precision == "int8" and calibration_dir is None:
        raise ValueError("INT8 TensorRT builds require a calibration directory")
    if workspace_mb <= 0:
        raise ValueError("workspace_mb must be positive")

    onnx = Path(onnx_path)
    engine = Path(engine_path)
    calibration = Path(calibration_dir).resolve() if calibration_dir is not None else None
    command = _build_command(shutil.which("trtexec") or "trtexec", onnx, engine, precision, calibration, workspace_mb)
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise RtxToolsUnavailable(
            "Local RTX TensorRT operation requires 'trtexec', which is unavailable. "
            "Install TensorRT and ensure trtexec is on PATH."
        ) from exc
    if not engine.exists():
        raise RuntimeError(f"TensorRT build completed but did not create engine: {engine}")

    resolved_engine = engine.resolve()
    return {
        "command": command,
        "precision": precision,
        "onnx_path": str(onnx.resolve()),
        "engine_path": str(resolved_engine),
        "engine_size_bytes": resolved_engine.stat().st_size,
        "calibration_dir": str(calibration) if calibration is not None else None,
        "calibration_provenance": str(calibration) if calibration is not None else None,
        "workspace_mb": int(workspace_mb),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _parse_latency(output: str, percentile: int) -> float | None:
    patterns = (
        rf"percentile\({percentile}%\)\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms",
        rf"percentile\s+{percentile}%\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms",
    )
    for pattern in patterns:
        match = re.search(pattern, output, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def benchmark_tensorrt_engine(
    engine_path: Path,
    iterations: int = 100,
    warmup: int = 20,
    device_label: str = "rtx3070",
) -> dict[str, Any]:
    """Benchmark a local TensorRT engine and normalize ``trtexec`` timing output."""

    engine = Path(engine_path)
    if not engine.exists():
        raise FileNotFoundError(engine)
    if iterations <= 0 or warmup < 0:
        raise ValueError("iterations must be positive and warmup cannot be negative")

    command = [
        _require_trtexec(),
        f"--loadEngine={engine}",
        f"--iterations={int(iterations)}",
        f"--warmUp={int(warmup)}",
        "--noDataTransfers",
        "--verbose",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    output = completed.stdout + "\n" + completed.stderr
    median_match = re.search(
        r"Latency:\s*min\s*=\s*[^,]+,\s*max\s*=\s*[^,]+,\s*mean\s*=\s*[^,]+,\s*median\s*=\s*([0-9.]+)",
        output,
        flags=re.IGNORECASE,
    )
    if not median_match:
        raise RuntimeError("Could not parse TensorRT latency from trtexec output.")
    median_ms = float(median_match.group(1))
    return {
        "command": command,
        "engine_path": str(engine.resolve()),
        "engine_size_bytes": engine.stat().st_size,
        "device_label": device_label,
        "iterations": int(iterations),
        "warmup": int(warmup),
        "latency_p50_ms": median_ms,
        "latency_p95_ms": _parse_latency(output, 95),
        "fps": 1000.0 / median_ms if median_ms else None,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
