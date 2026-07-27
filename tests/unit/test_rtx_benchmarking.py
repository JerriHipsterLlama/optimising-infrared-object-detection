from __future__ import annotations

import subprocess

import pytest

from infrared_detection.benchmarking.rtx import (
    RtxToolsUnavailable,
    benchmark_tensorrt_engine,
    build_tensorrt_engine,
)


def _completed_trtexec(stdout: str = ""):
    return subprocess.CompletedProcess(["trtexec"], 0, stdout=stdout, stderr="")


def test_tensorrt_command_uses_fp16_flag(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: commands.append(command) or _completed_trtexec())
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"engine")

    build_tensorrt_engine(tmp_path / "model.onnx", engine_path, "fp16", None, 1024)

    assert "--fp16" in commands[0]


def test_int8_build_requires_calibration_cache(tmp_path):
    with pytest.raises(ValueError, match="calibration"):
        build_tensorrt_engine(tmp_path / "model.onnx", tmp_path / "model.engine", "int8", None, 1024)


@pytest.mark.parametrize("cache_name", ["calibration", "missing.cache"])
def test_int8_build_requires_existing_regular_calibration_cache(tmp_path, cache_name):
    calibration_directory = tmp_path / "calibration"
    calibration_directory.mkdir()
    calibration_cache = calibration_directory if cache_name == "calibration" else tmp_path / cache_name

    with pytest.raises(ValueError, match="regular calibration cache file"):
        build_tensorrt_engine(
            tmp_path / "model.onnx",
            tmp_path / "model.engine",
            "int8",
            calibration_cache,
            1024,
        )


def test_tensorrt_build_records_command_calibration_cache_and_engine_size(monkeypatch, tmp_path):
    commands = []
    calibration_cache = tmp_path / "calibration.cache"
    calibration_cache.write_bytes(b"calibration-cache")
    engine_path = tmp_path / "model.engine"

    def run(command, **kwargs):
        commands.append(command)
        engine_path.write_bytes(b"built-engine")
        return _completed_trtexec("engine built")

    monkeypatch.setattr(subprocess, "run", run)
    result = build_tensorrt_engine(tmp_path / "model.onnx", engine_path, "int8", calibration_cache, 2048)

    assert "--int8" in commands[0]
    assert f"--calib={calibration_cache.resolve()}" in commands[0]
    assert result["command"] == commands[0]
    assert result["calibration_cache"] == str(calibration_cache.resolve())
    assert result["calibration_cache_provenance"] == str(calibration_cache.resolve())
    assert result["engine_size_bytes"] == len(b"built-engine")


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
