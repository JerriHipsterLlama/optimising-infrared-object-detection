from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import onnx
import cv2
import pytest
from onnx import TensorProto, helper

from infrared_detection.compression.quantization.onnx_precision import (
    CalibrationImageReader,
    prepare_precision_onnx,
)


def _onnx_model(path: Path, *, height: int = 4, width: int = 6) -> Path:
    input_info = helper.make_tensor_value_info(
        "images", TensorProto.FLOAT, [1, 3, height, width]
    )
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])
    node = helper.make_node("ReduceMean", ["images"], ["output"], keepdims=0)
    model = helper.make_model(helper.make_graph([node], "test", [input_info], [output_info]))
    onnx.save(model, path)
    return path


def test_fp32_reuses_source_graph(tmp_path):
    source = _onnx_model(tmp_path / "source.onnx")
    output = tmp_path / "fp32" / "model.onnx"

    metadata = prepare_precision_onnx(source, output, "fp32")

    assert output.read_bytes() == source.read_bytes()
    assert metadata["precision"] == "fp32"
    assert metadata["converter"] == "copy"


def test_fp16_calls_modelopt_autocast_and_checks_output(tmp_path, monkeypatch):
    source = _onnx_model(tmp_path / "source.onnx")
    output = tmp_path / "fp16" / "model.onnx"
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        Path(command[command.index("--output_path") + 1]).write_bytes(b"fp16")
        return types.SimpleNamespace(stdout="converted", stderr="")

    monkeypatch.setattr("infrared_detection.compression.quantization.onnx_precision.subprocess.run", fake_run)

    metadata = prepare_precision_onnx(source, output, "fp16")

    assert output.read_bytes() == b"fp16"
    assert calls[0][0][1:4] == ["-m", "modelopt.onnx.autocast", "--onnx_path"]
    assert metadata["converter"] == "modelopt.onnx.autocast"

    monkeypatch.setattr(
        "infrared_detection.compression.quantization.onnx_precision.subprocess.run",
        lambda *args, **kwargs: types.SimpleNamespace(stdout="", stderr=""),
    )
    with pytest.raises(FileNotFoundError, match="did not write output"):
        prepare_precision_onnx(source, tmp_path / "missing.onnx", "fp16")


def test_int8_uses_streaming_calibration_reader(tmp_path, monkeypatch):
    source = _onnx_model(tmp_path / "source.onnx")
    output = tmp_path / "int8" / "model.onnx"
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    cv2.imwrite(str(calibration / "one.jpg"), np.zeros((4, 6, 3), dtype=np.uint8))
    cv2.imwrite(str(calibration / "two.jpg"), np.ones((4, 6, 3), dtype=np.uint8))
    reader = CalibrationImageReader(calibration, input_name="images", image_size=(4, 6))
    received = {}

    def fake_quantize(model, **kwargs):
        received.update(kwargs)
        assert received["calibration_data_reader"].get_next()["images"].shape == (1, 3, 4, 6)
        assert received["calibration_data_reader"].get_next()["images"].shape == (1, 3, 4, 6)
        assert received["calibration_data_reader"].get_next() is None
        Path(kwargs["output_path"]).write_bytes(b"int8")

    package = types.ModuleType("modelopt")
    onnx_package = types.ModuleType("modelopt.onnx")
    quantization = types.ModuleType("modelopt.onnx.quantization")
    quantization.quantize = fake_quantize
    monkeypatch.setitem(sys.modules, "modelopt", package)
    monkeypatch.setitem(sys.modules, "modelopt.onnx", onnx_package)
    monkeypatch.setitem(sys.modules, "modelopt.onnx.quantization", quantization)

    metadata = prepare_precision_onnx(
        source, output, "int8", calibration_reader=reader, calibration_method="entropy"
    )

    assert output.read_bytes() == b"int8"
    assert received["quantize_mode"] == "int8"
    assert received["calibration_method"] == "entropy"
    assert metadata["calibration_sample_count"] == 2
    assert metadata["calibration_source"] == str(calibration.resolve())
    assert metadata["input_path"] == str(source.resolve())


def test_calibration_paths_cannot_resolve_to_test_split(tmp_path):
    source = _onnx_model(tmp_path / "source.onnx")
    calibration = tmp_path / "dataset" / "images" / "test"
    calibration.mkdir(parents=True)

    with pytest.raises(ValueError, match="test"):
        CalibrationImageReader(calibration, input_name="images", image_size=(4, 6))
    with pytest.raises(ValueError, match="test"):
        CalibrationImageReader(
            tmp_path / "images" / "train",
            input_name="images",
            image_size=(4, 6),
            calibration_split="test",
        )


def test_empty_calibration_data_fails_before_quantize(tmp_path, monkeypatch):
    source = _onnx_model(tmp_path / "source.onnx")
    calibration = tmp_path / "empty"
    calibration.mkdir()
    reader = CalibrationImageReader(calibration, input_name="images", image_size=(4, 6))
    called = False

    def fake_quantize(*args, **kwargs):
        nonlocal called
        called = True

    package = types.ModuleType("modelopt")
    onnx_package = types.ModuleType("modelopt.onnx")
    quantization = types.ModuleType("modelopt.onnx.quantization")
    quantization.quantize = fake_quantize
    monkeypatch.setitem(sys.modules, "modelopt", package)
    monkeypatch.setitem(sys.modules, "modelopt.onnx", onnx_package)
    monkeypatch.setitem(sys.modules, "modelopt.onnx.quantization", quantization)

    with pytest.raises(ValueError, match="empty"):
        prepare_precision_onnx(
            source, tmp_path / "int8.onnx", "int8", calibration_reader=reader
        )
    assert not called


def test_modelopt_error_is_preserved_in_variant_row(tmp_path):
    import yaml

    from infrared_detection.evaluation.compression_matrix import (
        CompressionMatrixAdapters,
        run_compression_matrix,
    )

    checkpoint = tmp_path / "dense.pt"
    checkpoint.write_bytes(b"checkpoint")
    config_path = tmp_path / "matrix.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"checkpoint": str(checkpoint)},
                "data": {"calibration_image_dir": str(tmp_path / "calibration")},
                "precisions": ["fp32", "fp16", "int8"],
                "pruning": {"enabled": False},
                "experiment": {"output_dir": str(tmp_path / "output")},
            }
        ),
        encoding="utf-8",
    )
    source = tmp_path / "source.onnx"
    source.write_bytes(b"onnx")

    def convert(source_path, precision, output, config):
        del source_path, config
        if precision == "int8":
            raise RuntimeError("modelopt calibration failed")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(precision.encode())
        return {"precision": precision}

    adapters = CompressionMatrixAdapters(
        build_pruned_checkpoint=lambda *args: checkpoint,
        export_checkpoint=lambda *args: source,
        convert_precision=convert,
    )

    rows = run_compression_matrix(config_path, adapters=adapters)

    int8_row = next(row for row in rows if row["precision"] == "int8")
    assert int8_row["status"] == "failed"
    assert int8_row["error"] == "RuntimeError: modelopt calibration failed"
