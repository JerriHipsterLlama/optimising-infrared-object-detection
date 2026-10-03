from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from infrared_detection.compression.pruning import cluster_probe as cluster_probe_module
from infrared_detection.evaluation import cluster_workflow


def _metrics(map50_95: float) -> dict[str, float]:
    return {"map50": 0.6, "map50_95": map50_95, "precision": 0.7, "recall": 0.5}


def _screening_config(tmp_path: Path) -> Path:
    path = tmp_path / "cluster-screening.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model": {"checkpoint": "models/baseline.pt"},
                "data": {"dataset_yaml": "data/camel.yaml"},
                "experiment": {
                    "image_size": 336,
                    "num_classes": 4,
                    "seed": 7,
                    "batch_size": 1,
                    "output_dir": str(tmp_path / "screening-artifacts"),
                },
                "runtime": {"device": "cpu", "precision": "fp32"},
                "pruning": {
                    "candidate_layers": ["model.1.conv", "model.2.conv"],
                    "cluster_sizes": [8, 16],
                    "probe_ratios": [0.1, 0.2],
                    "protected_layers": ["model.22"],
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _screening_adapters(*, baseline_error: Exception | None = None, failed_probe: str | None = None):
    from infrared_detection.evaluation.cluster_workflow import ClusterScreeningAdapters

    def evaluate(model, config, device):
        del config, device
        if model == "baseline" and baseline_error:
            raise baseline_error
        return _metrics(0.5 if model == "baseline" else 0.49)

    def make_probe(model, layer, cluster_size, ratio):
        del model, cluster_size, ratio
        if layer == failed_probe:
            raise ValueError("unsupported dependency group")
        return layer

    return ClusterScreeningAdapters(
        load_model=lambda checkpoint: "baseline",
        evaluate=evaluate,
        safe_layers=lambda model, config: list(config["pruning"]["candidate_layers"]),
        make_probe=make_probe,
        stats=lambda model: {"parameter_count": 10 if model == "baseline" else 8},
    )


def test_planned_rows_include_only_baseline_and_configured_probes(tmp_path):
    rows = cluster_workflow.run_cluster_screening(_screening_config(tmp_path), dry_run=True)

    assert [row["stage"] for row in rows].count("baseline") == 1
    assert len([row for row in rows if row["stage"] == "probe"]) == 8
    assert {row["layer"] for row in rows if row["stage"] == "probe"} == {
        "model.1.conv",
        "model.2.conv",
    }
    assert {row["cluster_size"] for row in rows if row["stage"] == "probe"} == {8, 16}
    assert all(row["stage"] != "global" for row in rows)
    assert (tmp_path / "screening-artifacts" / "candidates.csv").exists()
    assert (tmp_path / "screening-artifacts" / "manifest.json").exists()


def test_cluster_screening_records_metrics_without_deployment_fields(tmp_path):
    rows = cluster_workflow.run_cluster_screening(
        _screening_config(tmp_path), adapters=_screening_adapters()
    )
    probes = [row for row in rows if row["stage"] == "probe"]
    assert len(probes) == 8
    assert all(row["status"] == "completed" for row in probes)
    assert all(row["map50_95"] == 0.49 for row in probes)
    assert all(row["parameter_count_reduction"] == pytest.approx(0.2) for row in probes)
    assert all(row["candidate_id"] and row["layer"] and row["cluster_size"] for row in probes)
    assert all(
        not {"target_device", "profile_device", "hardware_benchmarked", "benchmark_path", "exported_path"}
        & row.keys()
        for row in rows
    )


def test_baseline_failure_skips_all_probes(tmp_path):
    rows = cluster_workflow.run_cluster_screening(
        _screening_config(tmp_path),
        adapters=_screening_adapters(baseline_error=RuntimeError("baseline unavailable")),
    )
    baseline = next(row for row in rows if row["stage"] == "baseline")
    assert baseline["status"] == "failed"
    assert "baseline unavailable" in baseline["error"]
    assert all(row["status"] == "skipped" for row in rows if row["stage"] == "probe")


def test_probe_failure_does_not_abort_remaining_rows(tmp_path):
    rows = cluster_workflow.run_cluster_screening(
        _screening_config(tmp_path), adapters=_screening_adapters(failed_probe="model.1.conv")
    )
    probes = [row for row in rows if row["stage"] == "probe"]
    assert any(row["status"] == "failed" for row in probes)
    assert any(row["status"] == "completed" for row in probes)
    assert len(probes) == 8


def test_invalid_screening_config_fails_before_loading_models(tmp_path):
    path = _screening_config(tmp_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["pruning"]["cluster_sizes"] = []
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    loaded = []
    adapters = _screening_adapters()
    adapters.load_model = lambda checkpoint: loaded.append(checkpoint)

    with pytest.raises(ValueError, match="cluster_sizes"):
        cluster_workflow.run_cluster_screening(path, adapters=adapters)
    assert loaded == []


def test_channel_importance_uses_physical_conv_weight_width_when_metadata_is_stale():
    from infrared_detection.compression.pruning import compute_channel_importance

    module = nn.Conv2d(3, 4, kernel_size=1)
    module.out_channels = 3
    assert compute_channel_importance(module)[""].shape == (4,)


class _ConvWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=1, bias=False)


class _C2fLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.cv1 = _ConvWrapper()


class _DetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList([nn.Identity(), nn.Identity(), _C2fLike()])


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
    config = yaml.safe_load(_screening_config(tmp_path).read_text(encoding="utf-8"))
    cluster_workflow._structural_probe(_importance_model(), "model.2.cv1.conv", 2, 0.25, config)

    assert [spec.prune_indices for spec in specs] == [(1, 2)]


def test_structural_probe_aligns_example_input_to_yolo_stride(monkeypatch, tmp_path):
    shapes = []

    def record_prune(model, example, spec):
        del spec
        shapes.append(tuple(example.shape))
        return model

    monkeypatch.setattr(cluster_probe_module, "run_structural_probe", record_prune)
    config = yaml.safe_load(_screening_config(tmp_path).read_text(encoding="utf-8"))
    cluster_workflow._structural_probe(_importance_model(), "model.2.cv1.conv", 2, 0.25, config)

    assert shapes == [(1, 3, 352, 352)]


def test_structural_probe_applies_requested_complete_clusters_sequentially(monkeypatch, tmp_path):
    specs = []

    def record_prune(model, example, spec):
        del example
        specs.append(spec)
        return model

    monkeypatch.setattr(cluster_probe_module, "run_structural_probe", record_prune)
    config = yaml.safe_load(_screening_config(tmp_path).read_text(encoding="utf-8"))
    cluster_workflow._structural_probe(_importance_model(), "model.2.cv1.conv", 2, 0.5, config)

    assert len(specs) == 2
    assert all(len(spec.prune_indices) == 2 for spec in specs)


def test_structural_probe_rejects_pruning_the_entire_layer(monkeypatch, tmp_path):
    monkeypatch.setattr(cluster_probe_module, "run_structural_probe", lambda model, example, spec: model)
    config = yaml.safe_load(_screening_config(tmp_path).read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="complete cluster"):
        cluster_workflow._structural_probe(_importance_model(), "model.2.cv1.conv", 8, 1.0, config)
