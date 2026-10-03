from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from infrared_detection.evaluation.compression_matrix import CompressionMatrixAdapters, run_compression_matrix


def _config(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "baseline.pt"
    checkpoint.write_bytes(b"external baseline")
    path = tmp_path / "matrix.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model": {"checkpoint": str(checkpoint)},
                "precisions": ["fp32", "fp16", "int8"],
                "pruning": {"enabled": False},
                "experiment": {"output_dir": str(tmp_path / "matrix-output")},
            }
        ),
        encoding="utf-8",
    )
    return path, checkpoint


def test_onnx_matrix_keeps_successful_precision_rows_when_one_conversion_fails(tmp_path):
    config_path, checkpoint = _config(tmp_path)
    calls = []

    def export(source_checkpoint, config, output_dir):
        assert source_checkpoint == checkpoint
        output_dir.mkdir(parents=True, exist_ok=True)
        source = output_dir / "baseline.onnx"
        source.write_bytes(b"base onnx")
        return source

    def convert(source, precision, output, config):
        calls.append(precision)
        if precision == "int8":
            raise RuntimeError("calibration failed")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(precision.encode())
        return {"precision": precision, "converter": "test"}

    adapters = CompressionMatrixAdapters(
        build_pruned_checkpoint=lambda *args: pytest.fail("pruning is disabled"),
        export_checkpoint=export,
        convert_precision=convert,
    )

    rows = run_compression_matrix(config_path, adapters=adapters)

    assert calls == ["fp32", "fp16", "int8"]
    assert [row["status"] for row in rows] == ["completed", "completed", "failed"]
    assert rows[2]["error"] == "RuntimeError: calibration failed"
    assert all("engine_path" not in row and "latency_p50_ms" not in row for row in rows)
    manifest = json.loads((tmp_path / "matrix-output" / "manifest.json").read_text(encoding="utf-8"))
    assert [row["status"] for row in manifest["rows"]] == ["completed", "completed", "failed"]
