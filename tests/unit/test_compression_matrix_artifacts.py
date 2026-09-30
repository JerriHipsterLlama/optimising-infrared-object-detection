from pathlib import Path

import yaml

from infrared_detection.evaluation.compression_matrix import (
    CompressionMatrixAdapters,
    run_compression_matrix,
)


def _matrix_config(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "dense.pt"
    checkpoint.write_bytes(b"dense checkpoint")
    config_path = tmp_path / "matrix.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"checkpoint": str(checkpoint), "architecture": "yolov8n"},
                "data": {
                    "dataset_yaml": str(tmp_path / "dataset.yaml"),
                    "calibration_image_dir": str(tmp_path / "calibration"),
                },
                "precisions": ["fp32", "fp16", "int8"],
                "pruning": {
                    "enabled": True,
                    "candidate_layers": ["model.1.conv"],
                    "cluster_size": 8,
                    "prune_ratios": [0.1, 0.2, 0.3, 0.4, 0.5],
                    "allowed_map50_95_drop": 0.02,
                    "importance": "minimum_weight",
                },
                "experiment": {"output_dir": str(tmp_path / "artifacts"), "image_size": 352},
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _adapters(tmp_path: Path, *, fail_int8_for: str | None = None):
    exports: list[str] = []
    conversions: list[tuple[str, str]] = []
    prunes: list[float] = []

    def build_pruned_checkpoint(config, output_dir, ratio):
        del config
        prunes.append(ratio)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = output_dir / "pruned.pt"
        checkpoint.write_bytes(f"pruned:{ratio}".encode())
        return checkpoint

    def export_checkpoint(checkpoint, config, output_dir):
        del config
        candidate_id = output_dir.relative_to(tmp_path / "artifacts").as_posix()
        exports.append(candidate_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        source = output_dir / "source.onnx"
        source.write_bytes(b"base onnx")
        return source

    def convert_precision(source_onnx, precision, output_onnx, config):
        del config
        candidate_id = output_onnx.parents[1].name
        conversions.append((candidate_id, precision))
        if precision == "int8" and candidate_id == fail_int8_for:
            raise RuntimeError("mock precision conversion failed")
        output_onnx.parent.mkdir(parents=True, exist_ok=True)
        output_onnx.write_bytes(source_onnx.read_bytes() + precision.encode())
        return {"converter": "test-adapter", "precision": precision}

    adapters = CompressionMatrixAdapters(
        build_pruned_checkpoint=build_pruned_checkpoint,
        export_checkpoint=export_checkpoint,
        convert_precision=convert_precision,
    )
    return adapters, exports, conversions, prunes


def test_compression_exports_base_onnx_once_per_candidate(tmp_path):
    adapters, exports, conversions, prunes = _adapters(tmp_path)

    rows = run_compression_matrix(_matrix_config(tmp_path), adapters=adapters)

    assert len(rows) == 18
    assert len(exports) == 6
    assert len(set(exports)) == 6
    assert len(prunes) == 5
    assert len(conversions) == 18
    assert all(row["status"] == "completed" for row in rows)


def test_precision_outputs_use_deterministic_candidate_directories(tmp_path):
    adapters, _exports, _conversions, _prunes = _adapters(tmp_path)

    rows = run_compression_matrix(_matrix_config(tmp_path), adapters=adapters)

    by_id = {row["variant_id"]: row for row in rows}
    expected = tmp_path / "artifacts" / "cluster-8-ratio-0.1" / "fp16" / "model.onnx"
    assert Path(by_id["cluster-8-ratio-0.1-fp16"]["onnx_path"]) == expected.resolve()
    assert expected.is_file()
    assert (tmp_path / "artifacts" / "dense" / "fp32" / "model.onnx").is_file()


def test_compression_does_not_build_engines_or_evaluate_models(tmp_path):
    adapters, _exports, _conversions, _prunes = _adapters(tmp_path)

    rows = run_compression_matrix(_matrix_config(tmp_path), adapters=adapters)

    assert all("engine_path" not in row for row in rows)
    assert all("latency_p50_ms" not in row for row in rows)
    assert all("map50_95" not in row for row in rows)
    assert not hasattr(adapters, "build_engine")
    assert not hasattr(adapters, "evaluate_engine")
    assert not hasattr(adapters, "benchmark_engine")


def test_precision_conversion_failure_preserves_sibling_rows(tmp_path):
    adapters, _exports, _conversions, _prunes = _adapters(tmp_path, fail_int8_for="dense")

    rows = run_compression_matrix(_matrix_config(tmp_path), adapters=adapters)

    dense_rows = {row["precision"]: row for row in rows if row["candidate_id"] == "dense"}
    assert dense_rows["fp32"]["status"] == "completed"
    assert dense_rows["fp16"]["status"] == "completed"
    assert dense_rows["int8"]["status"] == "failed"
    assert "mock precision conversion failed" in dense_rows["int8"]["error"]
    assert len(rows) == 18
    assert all(row["status"] == "completed" for row in rows if row["candidate_id"] != "dense")
