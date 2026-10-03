from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import cv2
import pytest
from onnx import TensorProto, helper
from onnx import numpy_helper

from infrared_detection.compression.quantization.onnx_precision import (
    CalibrationImageReader,
    prepare_precision_onnx,
)


def _onnx_model(path: Path, *, height: int = 4, width: int = 6) -> Path:
    input_info = helper.make_tensor_value_info(
        "images", TensorProto.FLOAT, [1, 3, height, width]
    )
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, height, width])
    weights = numpy_helper.from_array(np.ones((2, 3, 1, 1), dtype=np.float32), "weights")
    bias = numpy_helper.from_array(np.zeros((2,), dtype=np.float32), "bias")
    node = helper.make_node("Conv", ["images", "weights", "bias"], ["output"])
    graph = helper.make_graph([node], "test", [input_info], [output_info], [weights, bias])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.save(model, path)
    return path


def test_fp32_reuses_source_graph(tmp_path):
    source = _onnx_model(tmp_path / "source.onnx")
    output = tmp_path / "fp32" / "model.onnx"

    metadata = prepare_precision_onnx(source, output, "fp32")

    assert output.read_bytes() == source.read_bytes()
    assert metadata["precision"] == "fp32"
    assert metadata["converter"] == "copy"


def test_fp16_conversion_keeps_external_io_float32(tmp_path):
    source = _onnx_model(tmp_path / "source.onnx")
    output = tmp_path / "fp16" / "model.onnx"

    metadata = prepare_precision_onnx(source, output, "fp16")
    converted = onnx.load(output)

    onnx.checker.check_model(converted)
    assert converted.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert converted.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert any(weight.data_type == TensorProto.FLOAT16 for weight in converted.graph.initializer)
    assert metadata["converter"] == "onnxconverter-common"


def test_int8_uses_qdq_and_records_calibration_provenance(tmp_path):
    source = _onnx_model(tmp_path / "source.onnx")
    output = tmp_path / "int8" / "model.onnx"
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    cv2.imwrite(str(calibration / "one.jpg"), np.zeros((4, 6, 3), dtype=np.uint8))
    cv2.imwrite(str(calibration / "two.jpg"), np.ones((4, 6, 3), dtype=np.uint8))
    reader = CalibrationImageReader(calibration, input_name="images", image_size=(4, 6))
    metadata = prepare_precision_onnx(
        source, output, "int8", calibration_reader=reader, calibration_method="entropy"
    )
    converted = onnx.load(output)

    onnx.checker.check_model(converted)
    assert {node.op_type for node in converted.graph.node} >= {"QuantizeLinear", "DequantizeLinear"}
    assert metadata["converter"] == "onnxruntime.quantization.quantize_static"
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


def test_empty_calibration_data_fails_before_quantize(tmp_path):
    source = _onnx_model(tmp_path / "source.onnx")
    calibration = tmp_path / "empty"
    calibration.mkdir()
    reader = CalibrationImageReader(calibration, input_name="images", image_size=(4, 6))
    output = tmp_path / "int8.onnx"
    with pytest.raises(ValueError, match="empty"):
        prepare_precision_onnx(
            source, output, "int8", calibration_reader=reader
        )
    assert not output.exists()


def test_precision_conversion_error_is_preserved_in_variant_row(tmp_path):
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
            raise RuntimeError("onnxruntime calibration failed")
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
    assert int8_row["error"] == "RuntimeError: onnxruntime calibration failed"
