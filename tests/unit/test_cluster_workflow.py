from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from infrared_detection import export as export_module
from infrared_detection.compression.pruning import cluster_probe as cluster_probe_module
from infrared_detection.evaluation import cluster_workflow
from infrared_detection.evaluation.cluster_workflow import (
    ClusterEvaluationAdapters,
    _planned_filterwise_rows,
    merge_jetson_metrics,
    run_cluster_evaluation,
    run_filterwise_evaluation,
)


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
                "targets": {
                    "rtx_screening_device": "rtx",
                    "orin_target": "jetson_orin_nano",
                    "orin_execution_device": "0",
                },
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
        make_filterwise_step=lambda model, layer, config: {"kind": "probe", "layer": layer},
        save_checkpoint=lambda model, output_dir: _write_fixture_checkpoint(output_dir),
    )


def _write_fixture_export(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / "candidate.onnx"
    artifact.write_bytes(b"x" * (4 if output_dir.name == "baseline" else 2))
    return artifact


def _write_fixture_checkpoint(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "candidate.pt"
    checkpoint.write_bytes(b"checkpoint")
    return checkpoint


def test_dry_run_emits_baseline_probe_and_global_manifest(tmp_path):
    rows = run_cluster_evaluation(write_config(tmp_path), dry_run=True)

    assert [row["stage"] for row in rows] == ["baseline", "probe", "probe", "global", "global"]
    assert all("candidate_id" in row for row in rows)
    assert len({row["candidate_id"] for row in rows}) == len(rows)
    assert (tmp_path / "artifacts" / "candidates.csv").exists()
    assert (tmp_path / "artifacts" / "manifest.json").exists()


def test_filterwise_planner_removes_every_count_without_cluster_grouping():
    config = {
        "pruning": {
            "filter_sweep_layers": ["model.0.conv"],
            "filter_sweep_widths": {"model.0.conv": 4},
        }
    }

    rows = _planned_filterwise_rows(config)

    assert [row["stage"] for row in rows] == ["baseline", "filterwise", "filterwise", "filterwise"]
    assert [row["filters_removed"] for row in rows[1:]] == [1, 2, 3]
    assert [row["filters_after"] for row in rows[1:]] == [3, 2, 1]
    assert [row["filter_reduction"] for row in rows[1:]] == [0.25, 0.5, 0.75]
    assert [row["candidate_id"] for row in rows[1:]] == [
        "filterwise-model-0-conv-filters-1",
        "filterwise-model-0-conv-filters-2",
        "filterwise-model-0-conv-filters-3",
    ]


def test_filterwise_planner_requires_a_width_for_each_layer():
    with pytest.raises(ValueError, match="filter_sweep_widths"):
        _planned_filterwise_rows(
            {"pruning": {"filter_sweep_layers": ["model.0.conv"], "filter_sweep_widths": {}}}
        )


def test_filterwise_auto_planner_uses_runtime_layer_widths_in_order():
    rows = _planned_filterwise_rows(
        {"pruning": {"filter_sweep_layers": "auto"}},
        layer_widths={"model.2": 3, "model.1": 2},
    )

    assert [(row["layer"], row["filters_removed"]) for row in rows[1:]] == [
        ("model.2", 1),
        ("model.2", 2),
        ("model.1", 1),
    ]


def test_filterwise_auto_planner_defers_candidates_until_runtime_widths_are_available():
    rows = _planned_filterwise_rows({"pruning": {"filter_sweep_layers": "auto"}})

    assert [row["stage"] for row in rows] == ["baseline"]


def test_filterwise_layer_widths_reads_conv_output_channels():
    model = SimpleNamespace(model=nn.Sequential(nn.Conv2d(3, 4, kernel_size=1)))

    widths = cluster_workflow._filterwise_layer_widths(model, ["0"])

    assert widths == {"0": 4}


def test_filterwise_auto_workflow_expands_safe_layers_after_loading_baseline(monkeypatch, tmp_path, adapters):
    config = yaml.safe_load(write_config(tmp_path).read_text(encoding="utf-8"))
    config["pruning"]["filter_sweep_layers"] = "auto"
    config["pruning"].pop("filter_sweep_widths", None)
    config["export"] = {"format": "onnx"}
    config_path = tmp_path / "auto-filterwise.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    adapters.safe_layers = lambda model, config: ["model.2", "model.1"]
    monkeypatch.setattr(
        cluster_workflow,
        "_filterwise_layer_widths",
        lambda model, layers: {"model.2": 3, "model.1": 2},
    )

    rows = run_filterwise_evaluation(config_path, adapters=adapters)

    assert [(row["layer"], row["filters_removed"]) for row in rows[1:]] == [
        ("model.2", 1),
        ("model.2", 2),
        ("model.1", 1),
    ]


def test_filterwise_auto_workflow_resumes_completed_candidates(monkeypatch, tmp_path, adapters):
    config = yaml.safe_load(write_config(tmp_path).read_text(encoding="utf-8"))
    config["pruning"]["filter_sweep_layers"] = "auto"
    config["pruning"].pop("filter_sweep_widths", None)
    config["export"] = {"format": "onnx"}
    config_path = tmp_path / "auto-filterwise-resume.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    adapters.safe_layers = lambda model, config: ["model.1"]
    monkeypatch.setattr(cluster_workflow, "_filterwise_layer_widths", lambda model, layers: {"model.1": 3})
    evaluations = {"count": 0}

    def evaluate(model, config, device):
        del model, config, device
        evaluations["count"] += 1
        return _metrics(0.50)

    adapters.evaluate = evaluate

    run_filterwise_evaluation(config_path, adapters=adapters)
    completed_evaluations = evaluations["count"]
    run_filterwise_evaluation(config_path, adapters=adapters)

    assert evaluations["count"] == completed_evaluations


def test_filterwise_step_recomputes_minimum_weight_and_removes_one_filter(monkeypatch, tmp_path):
    model = _importance_model()
    config = yaml.safe_load(write_config(tmp_path).read_text(encoding="utf-8"))
    scores = iter(
        [
            {"model.2.cv1.conv": torch.tensor([9.0, 0.1, 0.2, 8.0, 0.3, 7.0, 6.0, 5.0])},
            {"model.2.cv1.conv": torch.tensor([9.0, 8.0, 0.05, 7.0, 6.0, 5.0, 4.0, 3.0])},
        ]
    )
    pruned_indices = []

    monkeypatch.setattr(
        "infrared_detection.compression.pruning.compute_channel_importance",
        lambda model: next(scores),
    )
    monkeypatch.setattr(
        cluster_probe_module,
        "run_filterwise_probe",
        lambda model, example_input, layer_name, prune_indices: (
            pruned_indices.append((layer_name, prune_indices)) or model
        ),
    )

    cluster_workflow._filterwise_step(model, "model.2.cv1.conv", config)
    cluster_workflow._filterwise_step(model, "model.2.cv1.conv", config)

    assert pruned_indices == [
        ("model.2.cv1.conv", (1,)),
        ("model.2.cv1.conv", (2,)),
    ]


def test_filterwise_probe_returns_the_pruned_model(monkeypatch, tmp_path):
    model = _importance_model()
    config = yaml.safe_load(write_config(tmp_path).read_text(encoding="utf-8"))
    monkeypatch.setattr(
        cluster_probe_module,
        "run_filterwise_probe",
        lambda model, example_input, layer_name, prune_indices: model,
    )

    result = cluster_workflow._filterwise_probe(model, "model.2.cv1.conv", 1, config)

    assert result is model


def test_filterwise_rtx_config_selects_all_layers_and_stops_below_point_three():
    config = yaml.safe_load(
        Path("configs/experiments/filterwise_rtx_screening.yaml").read_text(encoding="utf-8")
    )

    assert config["pruning"]["filter_sweep_layers"] == "auto"
    assert "filter_sweep_widths" not in config["pruning"]
    assert config["screening"]["early_stop"] is True
    assert config["screening"]["early_stop_map50_95"] == 0.30
    assert config["screening"]["early_stop_consecutive"] == 1


def test_filterwise_workflow_profiles_all_candidates_without_serialized_size_gate(monkeypatch, tmp_path, adapters):
    config_path = write_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["pruning"]["safe_layers"] = ["model.1"]
    config["pruning"]["filter_sweep_layers"] = ["model.1"]
    config["pruning"]["filter_sweep_widths"] = {"model.1": 4}
    config["export"] = {"format": "onnx"}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    monkeypatch.setattr(
        cluster_workflow,
        "_filterwise_probe",
        lambda model, layer, filters_removed, config: {
            "kind": "probe",
            "layer": layer,
            "filters_removed": filters_removed,
        },
    )
    adapters.export = lambda model, config, output_dir: _write_fixture_export(output_dir)

    rows = run_filterwise_evaluation(config_path, adapters=adapters)

    candidates = [row for row in rows if row["stage"] == "filterwise"]
    assert len(candidates) == 3
    assert all(row["status"] == "screened_in" for row in candidates)
    assert all(row["hardware_benchmarked"] is True for row in candidates)
    assert all(row["export_validation_status"] == "passed" for row in candidates)


def test_filterwise_workflow_checkpoints_after_baseline_and_each_candidate(monkeypatch, tmp_path, adapters):
    config_path = write_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["pruning"]["safe_layers"] = ["model.1"]
    config["pruning"]["filter_sweep_layers"] = ["model.1"]
    config["pruning"]["filter_sweep_widths"] = {"model.1": 3}
    config["export"] = {"format": "onnx"}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(
        cluster_workflow,
        "_filterwise_probe",
        lambda model, layer, filters_removed, config: {"kind": "probe"},
    )
    checkpoints = []
    monkeypatch.setattr(
        cluster_workflow,
        "_write_artifacts",
        lambda output_dir, resolved_config_path, rows: checkpoints.append(
            [(row["candidate_id"], row["status"]) for row in rows]
        ),
    )

    run_filterwise_evaluation(config_path, adapters=adapters)

    assert len(checkpoints) == 3
    assert checkpoints[0] == [("baseline", "completed"),
                              ("filterwise-model-1-filters-1", "planned"),
                              ("filterwise-model-1-filters-2", "planned")]
    assert all(status == "screened_in" for _, status in checkpoints[-1][1:])


def test_filterwise_workflow_resumes_completed_rows_from_checkpoint(monkeypatch, tmp_path, adapters):
    config_path = write_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["pruning"]["safe_layers"] = ["model.1"]
    config["pruning"]["filter_sweep_layers"] = ["model.1"]
    config["pruning"]["filter_sweep_widths"] = {"model.1": 3}
    config["export"] = {"format": "onnx"}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(
        cluster_workflow,
        "_filterwise_probe",
        lambda model, layer, filters_removed, config: {"kind": "probe"},
    )

    run_filterwise_evaluation(config_path, adapters=adapters)

    calls = {"evaluate": 0, "export": 0, "profile": 0}
    adapters.evaluate = lambda model, config, device: calls.__setitem__("evaluate", calls["evaluate"] + 1) or _metrics(0.50)
    adapters.export = lambda model, config, output_dir: calls.__setitem__("export", calls["export"] + 1) or _write_fixture_export(output_dir)
    adapters.profile = lambda exported_path, device: calls.__setitem__("profile", calls["profile"] + 1) or {"latency_p50_ms": 4.0}

    rows = run_filterwise_evaluation(config_path, adapters=adapters)

    assert rows[0]["status"] == "completed"
    assert all(row["status"] == "screened_in" for row in rows[1:])
    assert calls == {"evaluate": 0, "export": 0, "profile": 0}


def test_filterwise_workflow_prunes_sequentially_and_can_continue_full_curve(tmp_path, adapters):
    config_path = write_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["pruning"]["safe_layers"] = ["model.1"]
    config["pruning"]["filter_sweep_layers"] = ["model.1"]
    config["pruning"]["filter_sweep_widths"] = {"model.1": 4}
    config["screening"] = {
        "max_map50_95_drop": 0.02,
        "early_stop": True,
        "early_stop_map50_95": 0.001,
        "early_stop_consecutive": 1,
        "latency_profile_removals": [1],
    }
    config["export"] = {"format": "onnx"}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    seen_models = []

    def evaluate(model, config, device):
        del config, device
        seen_models.append(model)
        return _metrics(0.50)

    adapters.evaluate = evaluate

    rows = run_filterwise_evaluation(config_path, adapters=adapters, full_curve=True)

    candidates = [row for row in rows if row["stage"] == "filterwise"]
    assert len(candidates) == 3
    assert all(row["status"] == "screened_in" for row in candidates)
    assert len(seen_models) == 4
    assert all(row["checkpoint_path"].endswith("candidate.pt") for row in candidates)
    assert candidates[0]["hardware_benchmarked"] is True
    assert candidates[1]["hardware_benchmarked"] is False


def test_filterwise_workflow_evaluates_a_reloaded_candidate_copy(tmp_path, adapters):
    config_path = write_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["pruning"]["safe_layers"] = ["model.1"]
    config["pruning"]["filter_sweep_layers"] = ["model.1"]
    config["pruning"]["filter_sweep_widths"] = {"model.1": 4}
    config["screening"] = {
        "max_map50_95_drop": 0.02,
        "early_stop": False,
        "latency_profile_removals": [],
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    evaluated_models = []

    def evaluate(model, config, device):
        del config, device
        evaluated_models.append(model)
        model["evaluated"] = True
        return _metrics(0.50)

    def step(model, layer, config):
        del layer, config
        assert not model.get("evaluated", False)
        return model

    adapters.evaluate = evaluate
    adapters.make_filterwise_step = step

    rows = run_filterwise_evaluation(config_path, adapters=adapters, full_curve=True)

    assert len(evaluated_models) == 4
    assert all(model["evaluated"] for model in evaluated_models)
    assert all(row["status"] == "screened_in" for row in rows[1:])


def test_filterwise_workflow_early_stops_after_consecutive_near_zero_accuracy(tmp_path, adapters):
    config_path = write_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["pruning"]["safe_layers"] = ["model.1"]
    config["pruning"]["filter_sweep_layers"] = ["model.1"]
    config["pruning"]["filter_sweep_widths"] = {"model.1": 5}
    config["screening"] = {
        "max_map50_95_drop": 0.02,
        "early_stop": True,
        "early_stop_map50_95": 0.001,
        "early_stop_consecutive": 2,
        "latency_profile_removals": [],
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    accuracies = iter([0.50, 0.0, 0.0, 0.0])
    adapters.evaluate = lambda model, config, device: _metrics(next(accuracies))

    rows = run_filterwise_evaluation(config_path, adapters=adapters)

    candidates = [row for row in rows if row["stage"] == "filterwise"]
    assert [row["status"] for row in candidates[:2]] == ["screened_out", "screened_out"]
    assert all(row["status"] == "skipped" for row in candidates[2:])
    assert all("early stopping" in row["reason"].lower() for row in candidates[2:])


def test_filterwise_early_stop_requires_accuracy_strictly_below_threshold(tmp_path, adapters):
    config_path = write_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["pruning"]["safe_layers"] = ["model.1"]
    config["pruning"]["filter_sweep_layers"] = ["model.1"]
    config["pruning"]["filter_sweep_widths"] = {"model.1": 4}
    config["screening"] = {
        "early_stop": True,
        "early_stop_map50_95": 0.30,
        "early_stop_consecutive": 1,
        "latency_profile_removals": [],
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    accuracies = iter([0.50, 0.30, 0.29, 0.29])
    adapters.evaluate = lambda model, config, device: _metrics(next(accuracies))

    rows = run_filterwise_evaluation(config_path, adapters=adapters)

    candidates = [row for row in rows if row["stage"] == "filterwise"]
    assert [row["status"] for row in candidates] == ["screened_out", "screened_out", "skipped"]


def test_screen_only_uses_rtx_and_skips_global_candidates(tmp_path, adapters):
    config_path = write_config(tmp_path)
    evaluated_devices = []

    def evaluate(model, config, device):
        del model, config
        evaluated_devices.append(device)
        return _metrics(0.50)

    adapters.evaluate = evaluate
    rows = run_cluster_evaluation(config_path, adapters=adapters, screen_only=True)

    assert evaluated_devices[0] == "rtx"
    assert all(row["status"] == "skipped" for row in rows if row["stage"] == "global")
    assert all(row.get("hardware_benchmarked") is not True for row in rows)
    assert all(row["stage"] != "global" or "screen-only" in row["reason"] for row in rows)


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
    assert any(device == "jetson_orin_nano" for device in profiled_devices)


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
    assert baseline["metric_device"] == "0"
    assert baseline["target_device"] == "jetson_orin_nano"
    assert baseline["evaluation_device"] == "0"
    assert all(row["metric_device"] == "rtx" for row in probes)
    assert all(row["metric_device"] == "0" for row in globals_)
    assert all(row["target_device"] == "jetson_orin_nano" for row in globals_)
    assert all(row["evaluation_device"] == "0" for row in globals_)
    assert evaluated_devices[0] == "0"


def test_default_non_dry_workflow_refuses_orin_claim_off_target(monkeypatch, tmp_path):
    monkeypatch.setattr(cluster_workflow, "_is_jetson_orin_runtime", lambda: False)

    with pytest.raises(RuntimeError, match="remote Orin adapter|Jetson Orin Nano"):
        run_cluster_evaluation(write_config(tmp_path))


def test_topology_preserving_trainer_returns_exact_pruned_module():
    pruned = nn.Conv2d(3, 5, 1)

    class CompatibleDetectionTrainer:
        def get_model(self, cfg=None, weights=None, verbose=True):
            del cfg, weights, verbose
            return nn.Conv2d(3, 8, 1)

    trainer_type = cluster_workflow._topology_preserving_trainer(
        pruned,
        trainer_cls=CompatibleDetectionTrainer,
    )
    trainer = trainer_type.__new__(trainer_type)

    assert trainer.get_model(cfg="dense.yaml", weights=pruned, verbose=False) is pruned


def test_topology_preserving_trainer_fails_closed_for_incompatible_api():
    class IncompatibleDetectionTrainer:
        def get_model(self, architecture):
            del architecture

    with pytest.raises(RuntimeError, match="topology-preserving|Ultralytics"):
        cluster_workflow._topology_preserving_trainer(
            nn.Conv2d(3, 5, 1),
            trainer_cls=IncompatibleDetectionTrainer,
        )


def test_fine_tune_uses_topology_preserving_trainer_and_orin_execution_device(tmp_path):
    best = tmp_path / "fine-tune" / "weights" / "best.pt"

    class FakeYolo:
        def __init__(self):
            self.model = nn.Conv2d(3, 5, 1)
            self.trainer = None
            self.train_kwargs = None

        def train(self, **kwargs):
            self.train_kwargs = kwargs
            self.trainer = SimpleNamespace(best=best)
            return {"ok": True}

    model = FakeYolo()
    config = yaml.safe_load(write_config(tmp_path).read_text(encoding="utf-8"))

    assert cluster_workflow._fine_tune_yolo(model, config, tmp_path / "candidate") == best
    assert model.train_kwargs["device"] == "0"
    trainer_type = model.train_kwargs["trainer"]
    trainer = trainer_type.__new__(trainer_type)
    assert trainer.get_model(weights=model.model) is model.model


class _ConvWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)


class _C2fLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.cv1 = _ConvWrapper()


class _DetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList([nn.Identity(), nn.Identity(), _C2fLike()])


def test_safe_layers_accept_only_concrete_conv2d_paths(tmp_path):
    model = SimpleNamespace(model=_DetectionModel())
    config = yaml.safe_load(write_config(tmp_path).read_text(encoding="utf-8"))
    config["pruning"]["safe_layers"] = ["model.2.cv1.conv"]

    assert cluster_workflow._safe_layers(model, config) == ["model.2.cv1.conv"]

    config["pruning"]["safe_layers"] = ["model.2"]
    with pytest.raises(ValueError, match="Conv2d|prunable"):
        cluster_workflow._safe_layers(model, config)


def _importance_model() -> SimpleNamespace:
    detection = _DetectionModel()
    with torch.no_grad():
        conv = detection.model[2].cv1.conv
        for channel, magnitude in enumerate((9.0, 0.1, 0.2, 8.0, 0.3, 7.0, 6.0, 5.0)):
            conv.weight[channel].fill_(magnitude)
    return SimpleNamespace(model=detection)


def test_structural_probe_uses_lowest_importance_complete_cluster(monkeypatch, tmp_path):
    specs = []

    def record_prune(model, example, spec):
        del example
        specs.append(spec)
        return model

    monkeypatch.setattr(cluster_probe_module, "run_structural_probe", record_prune)
    config = yaml.safe_load(write_config(tmp_path).read_text(encoding="utf-8"))

    cluster_workflow._structural_probe(
        _importance_model(),
        "model.2.cv1.conv",
        cluster_size=2,
        ratio=0.25,
        config=config,
    )

    assert [spec.prune_indices for spec in specs] == [(1, 2)]


def test_structural_probe_aligns_dependency_graph_example_to_yolo_stride(monkeypatch, tmp_path):
    shapes = []

    def record_prune(model, example, spec):
        del spec
        shapes.append(tuple(example.shape))
        return model

    monkeypatch.setattr(cluster_probe_module, "run_structural_probe", record_prune)
    config = yaml.safe_load(write_config(tmp_path).read_text(encoding="utf-8"))

    cluster_workflow._structural_probe(
        _importance_model(),
        "model.2.cv1.conv",
        cluster_size=2,
        ratio=0.25,
        config=config,
    )

    assert shapes == [(1, 3, 352, 352)]


def test_structural_probe_applies_requested_complete_clusters_sequentially(monkeypatch, tmp_path):
    specs = []

    def record_prune(model, example, spec):
        del example
        specs.append(spec)
        return model

    monkeypatch.setattr(cluster_probe_module, "run_structural_probe", record_prune)
    config = yaml.safe_load(write_config(tmp_path).read_text(encoding="utf-8"))

    cluster_workflow._structural_probe(
        _importance_model(),
        "model.2.cv1.conv",
        cluster_size=2,
        ratio=0.50,
        config=config,
    )

    assert len(specs) == 2
    assert all(len(spec.prune_indices) == 2 for spec in specs)


def test_structural_probe_rejects_partial_cluster_that_would_remove_whole_layer(monkeypatch, tmp_path):
    monkeypatch.setattr(cluster_probe_module, "run_structural_probe", lambda model, example, spec: model)
    config = yaml.safe_load(write_config(tmp_path).read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="complete cluster"):
        cluster_workflow._structural_probe(
            _importance_model(),
            "model.2.cv1.conv",
            cluster_size=8,
            ratio=1.0,
            config=config,
        )


def test_append_row_allocates_unique_id_after_multiple_collisions():
    rows = []
    for _ in range(3):
        cluster_workflow._append_row(rows, cluster_workflow._base_row("duplicate", "probe"))

    assert [row["candidate_id"] for row in rows] == ["duplicate", "duplicate-2", "duplicate-3"]


def test_export_yolo_stages_checkpoint_and_artifact_under_candidate_directory(monkeypatch, tmp_path):
    checkpoint = tmp_path / "source" / "best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    adjacent_export = checkpoint.with_suffix(".onnx")
    adjacent_export.write_bytes(b"preserve-me")
    seen_sources = []

    def export_from_adjacent(model_path, format="onnx", **kwargs):
        del format, kwargs
        source = Path(model_path)
        seen_sources.append(source)
        artifact = source.with_suffix(".onnx")
        artifact.write_bytes(b"isolated-export")
        return artifact

    monkeypatch.setattr(export_module, "export_yolo", export_from_adjacent)
    config = yaml.safe_load(write_config(tmp_path).read_text(encoding="utf-8"))
    config["export"] = {"format": "onnx"}
    output_dir = tmp_path / "candidate"

    exported = cluster_workflow._export_yolo(checkpoint, config, output_dir)

    assert seen_sources[0].parent == output_dir
    assert exported.parent == output_dir
    assert exported.read_bytes() == b"isolated-export"
    assert adjacent_export.read_bytes() == b"preserve-me"


def test_export_yolo_saves_pruned_model_inside_candidate_before_export(monkeypatch, tmp_path):
    saved_paths = []

    class SaveablePrunedModel:
        def save(self, path):
            saved = Path(path)
            saved_paths.append(saved)
            saved.write_bytes(b"pruned-checkpoint")

    def export_from_adjacent(model_path, format="onnx", **kwargs):
        del format, kwargs
        artifact = Path(model_path).with_suffix(".onnx")
        artifact.write_bytes(b"pruned-export")
        return artifact

    monkeypatch.setattr(export_module, "export_yolo", export_from_adjacent)
    config = yaml.safe_load(write_config(tmp_path).read_text(encoding="utf-8"))
    config["export"] = {"format": "onnx"}
    output_dir = tmp_path / "probe"

    exported = cluster_workflow._export_yolo(SaveablePrunedModel(), config, output_dir)

    assert saved_paths[0].parent == output_dir
    assert exported.parent == output_dir
    assert exported.read_bytes() == b"pruned-export"


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
    assert all(row["export_validation_status"] == "failed" for row in probe_rows)
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
    assert all(row["export_validation_status"] == "failed" for row in probe_rows)
    assert all("format" in row["error"].lower() for row in probe_rows)


def test_default_profiler_rejects_orin_target_without_verified_jetson_runtime(monkeypatch, tmp_path):
    exported = tmp_path / "candidate.engine"
    exported.write_bytes(b"engine")
    monkeypatch.setattr(cluster_workflow, "_is_jetson_orin_runtime", lambda: False)

    with pytest.raises(RuntimeError, match="Refusing to label.*Orin"):
        cluster_workflow._profile_export(exported, "orin")


def test_manifest_waits_for_authoritative_orin_global_winner(tmp_path, adapters):
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
    adapters.validate_reduction = lambda before, after: {
        "parameter_count_reduction": 1.0 - float(after["parameter_count"]) / float(before["parameter_count"]),
        "serialized_reduction": 1.0 - float(after["serialized_bytes"]) / float(before["serialized_bytes"]),
    }

    rows = run_cluster_evaluation(write_config(tmp_path), adapters=adapters)

    manifest = json.loads((tmp_path / "artifacts" / "manifest.json").read_text(encoding="utf-8"))
    expected = "global-cluster-16-ratio-0.25"
    assert any(row["candidate_id"] == expected and row["status"] == "primary_feasible" for row in rows)
    assert all(row["export_validation_status"] == "passed" for row in rows if row["stage"] == "global")
    assert manifest["selected_candidate_ids"] == {"primary": None, "exploratory": None}


def test_merge_orin_metrics_updates_only_hardware_fields(tmp_path):
    rows = [
        {
            "candidate_id": "global-c4-r0.20",
            "stage": "global",
            "status": "primary_feasible",
            "export_validation_status": "passed",
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
            "device": "jetson_orin_nano",
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
    assert merged[0]["hardware_benchmarked"] is True
    assert merged[0]["profile_device"] == "jetson_orin_nano"
    assert merged[0]["benchmark_path"] == str(benchmark.resolve())
    assert merged[0]["benchmark_provenance"] == {
        "candidate_id": "global-c4-r0.20",
        "device": "jetson_orin_nano",
    }
    assert "8.1" in (tmp_path / "candidates.csv").read_text(encoding="utf-8")
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["selected_candidate_ids"] == {"primary": "global-c4-r0.20", "exploratory": None}
    assert manifest["benchmarks"] == [
        {
            "candidate_id": "global-c4-r0.20",
            "benchmark_path": str(benchmark.resolve()),
            "benchmark_provenance": {
                "candidate_id": "global-c4-r0.20",
                "device": "jetson_orin_nano",
            },
        }
    ]


@pytest.mark.parametrize("provenance", [{}, {"device": "rtx"}, {"target": "jetson_orin_nano_devkit"}])
def test_merge_jetson_metrics_rejects_missing_or_wrong_orin_provenance(tmp_path, provenance):
    rows = [{"candidate_id": "global-c4-r0.20", "stage": "global", "status": "failed"}]
    benchmark = write_json(
        tmp_path / "orin.json",
        {"candidate_id": "global-c4-r0.20", **provenance},
    )

    with pytest.raises(ValueError, match="jetson_orin_nano"):
        merge_jetson_metrics(rows, benchmark)


@pytest.mark.parametrize(
    "measurements",
    [
        {},
        {"latency_p50_ms": 8.1},
        {"latency_p50_ms": "8.1", "latency_p95_ms": 8.6},
        {"latency_p50_ms": 8.1, "latency_p95_ms": None},
        {"latency_p50_ms": True, "latency_p95_ms": 8.6},
        {"latency_p50_ms": float("nan"), "latency_p95_ms": 8.6},
    ],
)
def test_merge_jetson_metrics_requires_real_p50_and_p95_measurements(tmp_path, measurements):
    rows = [{"candidate_id": "global-c4-r0.20", "stage": "global", "hardware_benchmarked": False}]
    benchmark = write_json(
        tmp_path / "orin.json",
        {
            "candidate_id": "global-c4-r0.20",
            "device": "jetson_orin_nano",
            **measurements,
        },
    )

    with pytest.raises(ValueError, match="latency_p50_ms.*latency_p95_ms|latency_p95_ms.*latency_p50_ms"):
        merge_jetson_metrics(rows, benchmark)

    assert rows[0]["hardware_benchmarked"] is False


def test_manifest_does_not_select_unvalidated_export(tmp_path):
    rows = [
        {
            "candidate_id": "global-unvalidated",
            "stage": "global",
            "status": "primary_feasible",
            "map50_95": 0.5,
            "serialized_bytes": 1,
            "latency_p50_ms": 1.0,
            "latency_p95_ms": 2.0,
            "hardware_benchmarked": True,
            "export_validation_status": "not_run",
        }
    ]

    cluster_workflow._write_artifacts(tmp_path, tmp_path / "config.yaml", rows)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["selected_candidate_ids"] == {"primary": None, "exploratory": None}


def test_valid_orin_merge_reclassifies_failed_global_and_selects_only_benchmarked_rows(tmp_path):
    rows = [
        {"candidate_id": "baseline", "stage": "baseline", "map50_95": 0.50, "serialized_bytes": 1000},
        {
            "candidate_id": "global-failed",
            "stage": "global",
            "status": "failed",
            "map50_95": 0.495,
            "serialized_bytes": 500,
            "hardware_benchmarked": False,
            "export_validation_status": "passed",
        },
        {
            "candidate_id": "global-unbenchmarked",
            "stage": "global",
            "status": "primary_feasible",
            "map50_95": 0.499,
            "serialized_bytes": 1,
            "hardware_benchmarked": False,
            "export_validation_status": "passed",
        },
        {
            "candidate_id": "probe-feasible",
            "stage": "probe",
            "status": "primary_feasible",
            "map50_95": 0.50,
            "serialized_bytes": 1,
            "hardware_benchmarked": True,
            "export_validation_status": "passed",
        },
    ]
    benchmark = write_json(
        tmp_path / "orin.json",
        {
            "candidate_id": "global-failed",
            "device": "jetson_orin_nano",
            "latency_p50_ms": 8.1,
            "latency_p95_ms": 8.6,
        },
    )

    merged = merge_jetson_metrics(rows, benchmark)

    failed = next(row for row in merged if row["candidate_id"] == "global-failed")
    assert failed["status"] == "primary_feasible"
    assert failed["hardware_benchmarked"] is True
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["selected_candidate_ids"] == {"primary": "global-failed", "exploratory": None}
