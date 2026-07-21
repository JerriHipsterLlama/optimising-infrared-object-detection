from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from infrared_detection.evaluation import cluster_workflow
from infrared_detection.evaluation.cluster_workflow import ClusterEvaluationAdapters, run_cluster_evaluation


def write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "cluster-workflow.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"checkpoint": "models/baseline.pt"},
                "data": {"dataset_yaml": "data/camel.yaml"},
                "experiment": {
                    "image_size": 336,
                    "num_classes": 4,
                    "seed": 7,
                    "batch_size": 1,
                    "output_dir": str(tmp_path / "artifacts"),
                },
                "runtime": {"device": "rtx", "precision": "fp16", "conf": 0.25, "iou": 0.6},
                "pruning": {
                    "protected_layers": ["model.22"],
                    "safe_layers": ["model.1"],
                    "cluster_sizes": [8, 16],
                    "probe_ratios": [0.1],
                    "global_ratios": [0.25],
                    "fine_tune_epochs": 1,
                },
                "targets": {"rtx_screening_device": "rtx", "orin_target": "orin"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path


def _metrics(map50_95: float) -> dict[str, float]:
    return {"map50": 0.6, "map50_95": map50_95, "precision": 0.7, "recall": 0.5}


@pytest.fixture
def adapters(tmp_path: Path) -> ClusterEvaluationAdapters:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"checkpoint")
    exported = tmp_path / "candidate.onnx"
    exported.write_bytes(b"onnx")

    return ClusterEvaluationAdapters(
        load_model=lambda checkpoint_path: {"checkpoint": str(checkpoint_path)},
        evaluate=lambda model, config, device: _metrics(0.50),
        safe_layers=lambda model, config: list(config["pruning"]["safe_layers"]),
        make_probe=lambda model, layer, cluster_size, ratio: {"layer": layer, "cluster_size": cluster_size},
        make_global=lambda model, cluster_size, ratio: {"cluster_size": cluster_size, "ratio": ratio},
        fine_tune=lambda model, config, output_dir: checkpoint,
        reload_model=lambda checkpoint_path: {"checkpoint": str(checkpoint_path)},
        export=lambda checkpoint_path, config, output_dir: exported,
        profile=lambda exported_path, device: {"latency_p50_ms": 4.0, "latency_p95_ms": 5.0},
        stats=lambda model_or_path: {"parameter_count": 12, "serialized_bytes": 4},
    )


def test_dry_run_emits_baseline_probe_and_global_manifest(tmp_path):
    rows = run_cluster_evaluation(write_config(tmp_path), dry_run=True)

    assert [row["stage"] for row in rows] == ["baseline", "probe", "probe", "global", "global"]
    assert all("candidate_id" in row for row in rows)
    assert len({row["candidate_id"] for row in rows}) == len(rows)
    assert (tmp_path / "artifacts" / "candidates.csv").exists()
    assert (tmp_path / "artifacts" / "manifest.json").exists()


def test_one_failed_probe_does_not_stop_remaining_candidates(monkeypatch, tmp_path, adapters):
    calls = 0

    def fail_first_probe(model, layer, cluster_size, ratio):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("unsafe dependency group")
        return {"layer": layer, "cluster_size": cluster_size}

    monkeypatch.setattr(cluster_workflow, "run_structural_probe", fail_first_probe)
    adapters.make_probe = cluster_workflow.run_structural_probe

    rows = run_cluster_evaluation(write_config(tmp_path), adapters=adapters)

    assert any(row["status"] == "failed" for row in rows)
    assert any(row["stage"] == "global" for row in rows)
    assert len({row["candidate_id"] for row in rows}) == len(rows)


def test_successful_probes_are_profiled_on_rtx_before_global_orin_evaluation(tmp_path, adapters):
    profiled_devices = []

    def profile(exported_path, device):
        del exported_path
        profiled_devices.append(device)
        return {"latency_p50_ms": 4.0, "latency_p95_ms": 5.0}

    adapters.profile = profile

    rows = run_cluster_evaluation(write_config(tmp_path), adapters=adapters)

    probe_rows = [row for row in rows if row["stage"] == "probe"]
    assert all(row["screening_device"] == "rtx" for row in probe_rows)
    assert all(row["latency_p50_ms"] == 4.0 for row in probe_rows)
    assert profiled_devices.count("rtx") == len(probe_rows)
    assert any(device == "orin" for device in profiled_devices)
