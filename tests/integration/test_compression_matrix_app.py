from __future__ import annotations

import json
from pathlib import Path

from infrared_detection.evaluation.compression_matrix import (
    CompressionMatrixAdapters,
    run_compression_matrix,
)


def write_matrix_config(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "dense.pt"
    checkpoint.write_bytes(b"dense")
    evidence = tmp_path / "evidence.pt"
    evidence.write_bytes(b"evidence")
    manifest = tmp_path / "filterwise.json"
    manifest.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "layer": "model.2.conv",
                        "filters_removed": 8,
                        "status": "screened_in",
                        "checkpoint_path": str(evidence),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "matrix.yaml"
    config.write_text(
        "\n".join(
            [
                "model:",
                f"  checkpoint: {checkpoint}",
                "data:",
                "  dataset_yaml: data/camel/camel.yaml",
                "precisions: [fp32, fp16, int8]",
                "pruning:",
                "  enabled: true",
                f"  filterwise_manifest: {manifest}",
                "  evidence_layer: model.2.conv",
                "  evidence_filters_removed: 8",
                "  candidate_layers: [model.2.conv]",
                "  cluster_size: 8",
                "  prune_ratios: [0.05]",
                "  allowed_map50_95_drop: 0.02",
                "  importance: minimum_weight",
                "experiment:",
                "  image_size: 32",
                f"  output_dir: {tmp_path / 'results'}",
                "runtime:",
                "  device: 0",
                "  workspace_mb: 64",
            ]
        ),
        encoding="utf-8",
    )
    return config


def _fake_adapters(*, int8_fails: bool) -> CompressionMatrixAdapters:
    def build_pruned(config, output_dir, requested_ratio):
        del config, requested_ratio
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = output_dir / "pruned.pt"
        checkpoint.write_bytes(b"pruned")
        return checkpoint

    def export(checkpoint, config, output_dir):
        del config
        output_dir.mkdir(parents=True, exist_ok=True)
        onnx = output_dir / f"{checkpoint.stem}.onnx"
        onnx.write_bytes(b"onnx")
        return onnx

    def build_engine(onnx, engine, precision, calibration_cache, workspace_mb):
        del calibration_cache, workspace_mb
        if precision == "int8" and int8_fails:
            raise RuntimeError("INT8 calibration failed")
        engine.write_bytes(b"engine")
        return {"command": ["trtexec", f"--onnx={onnx}"], "engine_size_bytes": engine.stat().st_size}

    def evaluate(checkpoint, config, device):
        del checkpoint, config, device
        return {"map50": 0.8, "map50_95": 0.5, "precision": 0.75, "recall": 0.7}

    def benchmark(engine, device_label):
        return {
            "command": ["trtexec", f"--loadEngine={engine}"],
            "device_label": device_label,
            "latency_p50_ms": 2.0,
            "latency_p95_ms": 2.5,
            "fps": 500.0,
        }

    return CompressionMatrixAdapters(
        build_pruned_checkpoint=build_pruned,
        export_checkpoint=export,
        build_engine=build_engine,
        evaluate_checkpoint=evaluate,
        parameter_count=lambda checkpoint: len(checkpoint.read_bytes()),
        benchmark_engine=benchmark,
    )


def test_runner_keeps_successful_rows_when_int8_fails(tmp_path):
    rows = run_compression_matrix(write_matrix_config(tmp_path), adapters=_fake_adapters(int8_fails=True))

    assert any(row["status"] == "completed" and row["precision"] == "fp16" for row in rows), [
        (row["variant_id"], row["status"], row["error"]) for row in rows
    ]
    assert any(row["status"] == "failed" and row["precision"] == "int8" for row in rows)
    manifest = json.loads((tmp_path / "results" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["selected_candidate_id"] == "cluster-8-ratio-0.05-fp32"
