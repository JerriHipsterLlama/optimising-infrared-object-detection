from __future__ import annotations

import subprocess

import pytest

from infrared_detection.benchmarking.rtx import (
    RtxToolsUnavailable,
    benchmark_tensorrt_engine,
    build_tensorrt_engine,
    prepare_tensorrt_precision_onnx,
)


def _completed_trtexec(stdout: str = ""):
    return subprocess.CompletedProcess(["trtexec"], 0, stdout=stdout, stderr="")


def test_tensorrt_command_does_not_use_removed_fp16_flag(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: commands.append(command) or _completed_trtexec())
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"engine")

    build_tensorrt_engine(tmp_path / "model.onnx", engine_path, "fp16", None, 1024)

    assert "--fp16" not in commands[0]


def test_fp16_preparation_uses_modelopt_autocast(monkeypatch, tmp_path):
    commands = []
    onnx_path = tmp_path / "model.onnx"
    output_path = tmp_path / "model.fp16.onnx"
    onnx_path.write_bytes(b"onnx")

    def run(command, **kwargs):
        commands.append(command)
        output_path.write_bytes(b"autocast-onnx")
        return _completed_trtexec("autocast complete")

    monkeypatch.setattr(subprocess, "run", run)

    result = prepare_tensorrt_precision_onnx(onnx_path, output_path, "fp16")

    assert commands[0][1:4] == ["-m", "modelopt.onnx.autocast", "--onnx_path"]
    assert str(onnx_path) in commands[0]
    assert result["output_path"] == str(output_path.resolve())


def test_fp32_preparation_keeps_original_onnx(tmp_path):
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")

    result = prepare_tensorrt_precision_onnx(onnx_path, tmp_path / "unused.onnx", "fp32")

    assert result["output_path"] == str(onnx_path.resolve())


def test_fp16_preparation_reports_modelopt_failure(monkeypatch, tmp_path):
    onnx_path = tmp_path / "model.onnx"
    output_path = tmp_path / "model.fp16.onnx"
    onnx_path.write_bytes(b"onnx")

    def run(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr="modelopt unavailable")

    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(RuntimeError, match="ModelOpt FP16 autocast failed"):
        prepare_tensorrt_precision_onnx(onnx_path, output_path, "fp16")


def test_int8_build_is_deferred_for_tensorrt_11(tmp_path):
    with pytest.raises(ValueError, match="deferred"):
        build_tensorrt_engine(tmp_path / "model.onnx", tmp_path / "model.engine", "int8", None, 1024)


def test_int8_build_is_deferred_even_when_cache_is_supplied(tmp_path):
    calibration_cache = tmp_path / "calibration.cache"
    calibration_cache.write_bytes(b"calibration-cache")

    with pytest.raises(ValueError, match="deferred"):
        build_tensorrt_engine(tmp_path / "model.onnx", tmp_path / "model.engine", "int8", calibration_cache, 1024)


def test_tensorrt_benchmark_records_latency_percentile_and_fps(monkeypatch, tmp_path):
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"engine")
    commands = []
    output = (
        "Latency: min = 7.0 ms, max = 12.0 ms, mean = 8.0 ms, median = 7.5 ms, percentile(95%) = 10.5 ms\n"
    )
    monkeypatch.setattr("infrared_detection.benchmarking.rtx.shutil.which", lambda name: "trtexec")
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: commands.append(command) or _completed_trtexec(output))

    result = benchmark_tensorrt_engine(engine_path, iterations=10, warmup=2)

    assert commands[0][0] == "trtexec"
    assert result["device_label"] == "rtx3070"
    assert result["iterations"] == 10
    assert result["warmup"] == 2
    assert result["latency_p50_ms"] == 7.5
    assert result["latency_p95_ms"] == 10.5
    assert result["fps"] == pytest.approx(1000 / 7.5)


def test_tensorrt_benchmark_reports_clear_diagnostic_when_trtexec_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("infrared_detection.benchmarking.rtx.shutil.which", lambda name: None)
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"engine")

    with pytest.raises(RtxToolsUnavailable, match="trtexec"):
        benchmark_tensorrt_engine(engine_path)
