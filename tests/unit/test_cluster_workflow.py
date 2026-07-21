from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from infrared_detection.evaluation import cluster_workflow
from infrared_detection.evaluation.cluster_workflow import ClusterEvaluationAdapters, merge_jetson_metrics, run_cluster_evaluation


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


def write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _metrics(map50_95: float) -> dict[str, float]:
    return {"map50": 0.6, "map50_95": map50_95, "precision": 0.7, "recall": 0.5}


@pytest.fixture
def adapters(tmp_path: Path) -> ClusterEvaluationAdapters:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"checkpoint")
    exported = tmp_path / "candidate.onnx"
    exported.write_bytes(b"onnx")

    return ClusterEvaluationAdapters(
        load_model=lambda checkpoint_path: {"checkpoint": str(checkpoint_path), "kind": "loaded"},
        evaluate=lambda model, config, device: _metrics(0.50),
        safe_layers=lambda model, config: list(config["pruning"]["safe_layers"]),
        make_probe=lambda model, layer, cluster_size, ratio: {"layer": layer, "cluster_size": cluster_size, "kind": "probe"},
        make_global=lambda model, cluster_size, ratio: {"cluster_size": cluster_size, "ratio": ratio, "kind": "global"},
        fine_tune=lambda model, config, output_dir: checkpoint,
        reload_model=lambda checkpoint_path: {"checkpoint": str(checkpoint_path), "kind": "global"},
        export=lambda checkpoint_path, config, output_dir: _write_fixture_export(output_dir),
        profile=lambda exported_path, device: {"latency_p50_ms": 4.0, "latency_p95_ms": 5.0},
        stats=lambda model_or_path: {
            "parameter_count": 8 if isinstance(model_or_path, dict) and model_or_path.get("kind") in {"probe", "global"} else 10,
        },
    )


def _write_fixture_export(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / "candidate.onnx"
    artifact.write_bytes(b"x" * (4 if output_dir.name == "baseline" else 2))
    return artifact


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
        return {"layer": layer, "cluster_size": cluster_size, "kind": "probe"}

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


def test_baseline_and_global_metrics_preserve_authoritative_orin_provenance(tmp_path, adapters):
    evaluated_devices = []

    def evaluate(model, config, device):
        del model, config
        evaluated_devices.append(device)
        return {**_metrics(0.50), "metric_device": device}

    adapters.evaluate = evaluate

    rows = run_cluster_evaluation(write_config(tmp_path), adapters=adapters)

    baseline = next(row for row in rows if row["stage"] == "baseline")
    probes = [row for row in rows if row["stage"] == "probe"]
    globals_ = [row for row in rows if row["stage"] == "global"]
    assert baseline["metric_device"] == "orin"
    assert baseline["evaluation_device"] == "orin"
    assert all(row["metric_device"] == "rtx" for row in probes)
    assert all(row["metric_device"] == "orin" for row in globals_)
    assert all(row["evaluation_device"] == "orin" for row in globals_)
    assert evaluated_devices[0] == "orin"


def test_same_format_baseline_artifact_and_physical_reduction_are_required(tmp_path, adapters):
    export_calls = []

    def export(model_or_checkpoint, config, output_dir):
        del model_or_checkpoint
        export_calls.append(config["export"]["format"])
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact = output_dir / "candidate.onnx"
        artifact.write_bytes(b"same-size-artifact")
        return artifact

    config_path = write_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["export"] = {"format": "onnx"}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    adapters.export = export

    rows = run_cluster_evaluation(config_path, adapters=adapters)

    baseline = next(row for row in rows if row["stage"] == "baseline")
    probe_rows = [row for row in rows if row["stage"] == "probe"]
    assert baseline["exported_path"].endswith("candidate.onnx")
    assert baseline["artifact_format"] == "onnx"
    assert export_calls
    assert all(row["status"] in {"failed", "rejected_accuracy"} for row in probe_rows)
    assert all("serialized" in (row["error"] or "").lower() or "reduction" in (row["error"] or "").lower() for row in probe_rows)


def test_baseline_failure_retains_every_configured_candidate_with_skip_reason(tmp_path, adapters):
    config_path = write_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["pruning"]["probe_ratios"] = [0.1, 0.2]
    config["pruning"]["global_ratios"] = [0.25, 0.5]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def fail_baseline(model, config, device):
        del model, config, device
        raise RuntimeError("authoritative Orin baseline unavailable")

    adapters.evaluate = fail_baseline

    rows = run_cluster_evaluation(config_path, adapters=adapters)

    assert len(rows) == 1 + (1 * 2 * 2) + (2 * 2)
    assert rows[0]["status"] == "failed"
    assert all(row["status"] == "skipped" for row in rows[1:])
    assert all("baseline" in (row["error"] or "").lower() for row in rows[1:])


def test_dry_run_plans_each_probe_ratio_with_ratio_in_candidate_id(tmp_path):
    config_path = write_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["pruning"]["probe_ratios"] = [0.1, 0.2]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    rows = run_cluster_evaluation(config_path, dry_run=True)

    probe_ids = {row["candidate_id"] for row in rows if row["stage"] == "probe"}
    assert probe_ids == {"probe-model-1-cluster-8-ratio-0.1", "probe-model-1-cluster-8-ratio-0.2", "probe-model-1-cluster-16-ratio-0.1", "probe-model-1-cluster-16-ratio-0.2"}


def test_probe_screening_failure_retains_all_global_candidates_as_skipped(tmp_path, adapters):
    def fail_probe(model, layer, cluster_size, ratio):
        del model, layer, cluster_size, ratio
        raise ValueError("RTX screening unavailable")

    adapters.make_probe = fail_probe

    rows = run_cluster_evaluation(write_config(tmp_path), adapters=adapters)

    global_rows = [row for row in rows if row["stage"] == "global"]
    assert len(global_rows) == 2
    assert all(row["status"] == "skipped" for row in global_rows)
    assert all("no probe survived" in row["error"].lower() for row in global_rows)


def test_each_preplanned_row_is_resolved_when_safe_layers_or_candidate_shapes_repeat(tmp_path, adapters):
    config_path = write_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["pruning"]["safe_layers"] = ["model.1", "model.2"]
    config["pruning"]["cluster_sizes"] = [8, 8]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    adapters.safe_layers = lambda model, config: ["model.1"]

    rows = run_cluster_evaluation(config_path, adapters=adapters)

    assert len({row["candidate_id"] for row in rows}) == len(rows)
    assert all(row["status"] != "planned" for row in rows)
    skipped_probes = [
        row
        for row in rows
        if row["stage"] == "probe" and row["candidate_id"].startswith("probe-model-2-")
    ]
    assert skipped_probes
    assert all("safe-layer" in row["reason"] for row in skipped_probes)


def test_candidate_artifact_must_match_the_baseline_export_format(tmp_path, adapters):
    def export(model_or_checkpoint, config, output_dir):
        del model_or_checkpoint, config
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".onnx" if output_dir.name == "baseline" else ".engine"
        artifact = output_dir / f"candidate{suffix}"
        artifact.write_bytes(b"x" * (4 if output_dir.name == "baseline" else 2))
        return artifact

    adapters.export = export

    rows = run_cluster_evaluation(write_config(tmp_path), adapters=adapters)

    baseline = next(row for row in rows if row["stage"] == "baseline")
    probe_rows = [row for row in rows if row["stage"] == "probe"]
    assert baseline["artifact_format"] == "onnx"
    assert all(row["status"] == "failed" for row in probe_rows)
    assert all("format" in row["error"].lower() for row in probe_rows)


def test_default_profiler_rejects_orin_target_without_verified_jetson_runtime(monkeypatch, tmp_path):
    exported = tmp_path / "candidate.engine"
    exported.write_bytes(b"engine")
    monkeypatch.setattr(cluster_workflow, "_is_jetson_orin_runtime", lambda: False)

    with pytest.raises(RuntimeError, match="Refusing to label.*Orin"):
        cluster_workflow._profile_export(exported, "orin")


def test_manifest_records_the_smallest_authoritative_global_winner(tmp_path, adapters):
    checkpoints: dict[str, int] = {}

    def fine_tune(model, config, output_dir):
        del config
        checkpoint = output_dir / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        checkpoints[str(checkpoint)] = model["cluster_size"]
        return checkpoint

    def reload_model(checkpoint):
        return {"kind": "global", "cluster_size": checkpoints[str(checkpoint)]}

    def evaluate(model, config, device):
        del config
        if model.get("kind") == "global":
            return {**_metrics(0.499 if model["cluster_size"] == 16 else 0.495), "metric_device": device}
        return {**_metrics(0.50), "metric_device": device}

    def export(model_or_checkpoint, config, output_dir):
        del model_or_checkpoint, config
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact = output_dir / "candidate.onnx"
        if output_dir.name == "baseline":
            artifact.write_bytes(b"x" * 8)
        elif "cluster-16" in output_dir.name:
            artifact.write_bytes(b"x" * 2)
        else:
            artifact.write_bytes(b"x" * 4)
        return artifact

    adapters.fine_tune = fine_tune
    adapters.reload_model = reload_model
    adapters.evaluate = evaluate
    adapters.export = export

    rows = run_cluster_evaluation(write_config(tmp_path), adapters=adapters)

    manifest = json.loads((tmp_path / "artifacts" / "manifest.json").read_text(encoding="utf-8"))
    expected = "global-cluster-16-ratio-0.25"
    assert any(row["candidate_id"] == expected and row["status"] == "primary_feasible" for row in rows)
    assert manifest["selected_candidate_ids"] == {"primary": expected, "exploratory": None}


def test_merge_orin_metrics_updates_only_hardware_fields(tmp_path):
    rows = [
        {
            "candidate_id": "global-c4-r0.20",
            "stage": "global",
            "status": "primary_feasible",
            "map50_95": 0.495,
            "serialized_bytes": 1234,
            "latency_p50_ms": None,
            "latency_p95_ms": None,
            "fps": None,
            "peak_memory_mb": None,
            "power_w": None,
            "energy_mj_per_inference": None,
            "temperature_c": None,
        }
    ]
    benchmark = write_json(
        tmp_path / "orin.json",
        {
            "candidate_id": "global-c4-r0.20",
            "map50_95": 0.1,
            "serialized_bytes": 1,
            "latency_p50_ms": 8.1,
            "latency_p95_ms": 8.6,
            "fps": 123.4,
            "peak_memory_mb": 456.7,
            "power_w": 12.3,
            "energy_mj_per_inference": 99.6,
            "temperature_c": 48.2,
        },
    )

    merged = merge_jetson_metrics(rows, benchmark)

    assert merged[0]["map50_95"] == 0.495
    assert merged[0]["serialized_bytes"] == 1234
    assert merged[0]["latency_p50_ms"] == 8.1
    assert merged[0]["temperature_c"] == 48.2
    assert "8.1" in (tmp_path / "candidates.csv").read_text(encoding="utf-8")
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["selected_candidate_ids"] == {"primary": "global-c4-r0.20", "exploratory": None}
