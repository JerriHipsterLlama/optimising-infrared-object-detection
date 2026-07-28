import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import infrared_detection.evaluation.compression_matrix as compression_matrix
from infrared_detection.evaluation.compression_matrix import (
    build_pruned_checkpoint,
    load_compression_config,
    planned_variants,
    select_filterwise_candidate,
    write_compression_manifest,
)


def test_matrix_planner_creates_three_dense_and_three_rows_per_prune_ratio():
    rows = planned_variants(
        {
            "pruning": {
                "enabled": True,
                "filterwise_manifest": "artifacts/filterwise_rtx_screening/manifest.json",
                "evidence_layer": "model.8.cv2.conv",
                "candidate_layers": ["model.8.cv2.conv"],
                "cluster_size": 8,
                "evidence_filters_removed": 8,
                "prune_ratios": [0.05, 0.10],
                "allowed_map50_95_drop": 0.02,
                "importance": "minimum_weight",
            },
            "precisions": ["fp32", "fp16", "int8"],
        }
    )

    assert [row["variant_id"] for row in rows] == [
        "dense-fp32",
        "dense-fp16",
        "dense-int8",
        "cluster-8-ratio-0.05-fp32",
        "cluster-8-ratio-0.05-fp16",
        "cluster-8-ratio-0.05-int8",
        "cluster-8-ratio-0.1-fp32",
        "cluster-8-ratio-0.1-fp16",
        "cluster-8-ratio-0.1-int8",
    ]
    assert len(rows) == 9
    assert "cluster-8-ratio-0.05-fp16" in {row["variant_id"] for row in rows}
    assert all(row["status"] == "planned" for row in rows)
    assert all("provenance" in row for row in rows)
    pruned = rows[3]
    assert pruned["provenance"] == {
        "filterwise_manifest": "artifacts/filterwise_rtx_screening/manifest.json",
        "evidence_layer": "model.8.cv2.conv",
        "candidate_layers": ["model.8.cv2.conv"],
        "cluster_size": 8,
        "evidence_filters_removed": 8,
        "prune_ratio": 0.05,
        "allowed_map50_95_drop": 0.02,
        "importance": "minimum_weight",
    }


def test_planned_variants_reject_non_official_precision():
    with pytest.raises(ValueError, match="Precision matrix"):
        planned_variants({"precisions": ["fp32", "int4"]})


@pytest.mark.parametrize(
    "precisions",
    [["fp32"], ["fp32", "fp16", "int8", "int8"], ["int8", "fp16", "fp32"]],
)
def test_precision_matrix_requires_exact_ordered_official_set(precisions):
    with pytest.raises(ValueError, match="Precision matrix"):
        planned_variants({"precisions": precisions})


def test_precision_matrix_accepts_fp32_fp16_only():
    rows = planned_variants({"precisions": ["fp32", "fp16"]})

    assert [row["variant_id"] for row in rows] == ["dense-fp32", "dense-fp16"]


def test_load_compression_config_reads_yaml(tmp_path):
    config_path = tmp_path / "compression.yaml"
    config_path.write_text(
        "pruning:\n  enabled: false\nprecisions: [fp32, fp16, int8]\n",
        encoding="utf-8",
    )

    config = load_compression_config(config_path)

    assert config["precisions"] == ["fp32", "fp16", "int8"]
    assert config["_config_path"] == str(config_path.resolve())


def test_load_compression_config_accepts_fp32_fp16_matrix(tmp_path):
    config_path = tmp_path / "compression.yaml"
    config_path.write_text("precisions: [fp32, fp16]\n", encoding="utf-8")

    config = load_compression_config(config_path)

    assert config["precisions"] == ["fp32", "fp16"]


def test_rtx_config_declares_filterwise_evidence_without_inference():
    config_path = Path("configs/experiments/rtx_compression_matrix.yaml")

    config = load_compression_config(config_path)
    pruning = config["pruning"]

    assert pruning["filterwise_manifest"] == "runs/experiments/filterwise_rtx_screening/manifest.json"
    assert pruning["evidence_layer"] == "model.6.cv2.conv"
    assert pruning["candidate_layers"] == [
        "model.4.cv2.conv",
        "model.6.cv2.conv",
        "model.8.cv2.conv",
    ]
    assert pruning["cluster_size"] == 8
    assert pruning["evidence_filters_removed"] == 8
    assert pruning["prune_ratios"] == [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    assert pruning["allowed_map50_95_drop"] == 0.02
    assert pruning["importance"] == "minimum_weight"
    assert config["precisions"] == ["fp32", "fp16"]
    assert config["runtime"]["evaluation_device"] == "0"
    assert config["experiment"]["image_size"] == 352
    assert planned_variants(config)[3]["provenance"] == {
        "filterwise_manifest": pruning["filterwise_manifest"],
        "evidence_layer": pruning["evidence_layer"],
        "candidate_layers": pruning["candidate_layers"],
        "cluster_size": pruning["cluster_size"],
        "evidence_filters_removed": pruning["evidence_filters_removed"],
        "prune_ratio": 0.05,
        "allowed_map50_95_drop": pruning["allowed_map50_95_drop"],
        "importance": pruning["importance"],
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("candidate_layers", None, "candidate_layers"),
        ("cluster_size", 0, "cluster_size"),
        ("prune_ratios", [0.0], "prune_ratios"),
        ("allowed_map50_95_drop", -0.01, "allowed_map50_95_drop"),
        ("importance", "l2", "importance"),
    ],
)
def test_planned_variants_requires_explicit_cluster_pruning_contract(field, value, message):
    pruning = {
        "enabled": True,
        "filterwise_manifest": "filterwise.json",
        "evidence_layer": "model.2.conv",
        "evidence_filters_removed": 8,
        "candidate_layers": ["model.2.conv"],
        "cluster_size": 8,
        "prune_ratios": [0.05, 0.25],
        "allowed_map50_95_drop": 0.02,
        "importance": "minimum_weight",
    }
    if value is None:
        del pruning[field]
    else:
        pruning[field] = value

    with pytest.raises(ValueError, match=message):
        planned_variants({"pruning": pruning, "precisions": ["fp32", "fp16", "int8"]})


def test_manifest_writer_preserves_failed_and_successful_rows(tmp_path):
    write_compression_manifest(
        tmp_path,
        [
            {"variant_id": "dense-fp32", "status": "completed"},
            {"variant_id": "dense-fp16", "status": "failed", "error": "runtime"},
        ],
    )

    payload = json.loads((tmp_path / "manifest.json").read_text())
    assert payload["rows"][0]["status"] == "completed"
    assert payload["rows"][1]["status"] == "failed"
    with (tmp_path / "results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["status"] for row in rows} == {"completed", "failed"}


def test_manifest_writer_keeps_existing_rows_when_checkpoint_is_partial(tmp_path):
    write_compression_manifest(tmp_path, [{"variant_id": "dense-fp32", "status": "completed"}])
    write_compression_manifest(tmp_path, [{"variant_id": "dense-fp16", "status": "failed"}])

    payload = json.loads((tmp_path / "manifest.json").read_text())

    assert {row["variant_id"] for row in payload["rows"]} == {"dense-fp32", "dense-fp16"}


def test_select_filterwise_candidate_requires_matching_layer_and_count(tmp_path):
    manifest = tmp_path / "filterwise.json"
    manifest.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "layer": "model.2.cv2.conv",
                        "filters_removed": 8,
                        "status": "screened_in",
                        "checkpoint_path": "candidate.pt",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    row = select_filterwise_candidate(manifest, "model.2.cv2.conv", 8)

    assert row["checkpoint_path"] == "candidate.pt"


def test_select_filterwise_candidate_rejects_failed_row(tmp_path):
    manifest = tmp_path / "filterwise.json"
    manifest.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "layer": "model.2.cv2.conv",
                        "filters_removed": 8,
                        "status": "failed",
                        "checkpoint_path": "candidate.pt",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not usable"):
        select_filterwise_candidate(manifest, "model.2.cv2.conv", 8)


def test_build_pruned_checkpoint_prunes_dense_source_not_filterwise_evidence(
    monkeypatch, tmp_path
):
    source_checkpoint = tmp_path / "dense.pt"
    source_model = nn.Sequential(nn.Conv2d(3, 16, kernel_size=1), nn.Conv2d(16, 16, kernel_size=1))
    with torch.no_grad():
        source_model[0].weight.copy_(
            torch.arange(16, dtype=torch.float32).reshape(16, 1, 1, 1).repeat(1, 3, 1, 1)
        )
    torch.save(source_model, source_checkpoint)
    evidence_checkpoint = tmp_path / "filterwise-candidate.pt"
    torch.save(nn.Sequential(nn.Conv2d(3, 8, kernel_size=1)), evidence_checkpoint)
    manifest = tmp_path / "filterwise.json"
    manifest.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "layer": "0",
                        "filters_removed": 8,
                        "status": "screened_in",
                        "checkpoint_path": str(evidence_checkpoint),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    source_wrapper = SimpleNamespace(model=source_model)
    source_wrapper.save = lambda path: torch.save(source_wrapper.model, path)
    after_first_layer = nn.Sequential(nn.Conv2d(3, 8, kernel_size=1), nn.Conv2d(8, 16, kernel_size=1))
    with torch.no_grad():
        after_first_layer[1].weight.copy_(
            torch.arange(16, dtype=torch.float32).reshape(16, 1, 1, 1).repeat(1, 8, 1, 1)
        )
    final_model = nn.Sequential(nn.Conv2d(3, 8, kernel_size=1), nn.Conv2d(8, 8, kernel_size=1))

    def load_yolo(checkpoint_path):
        if Path(checkpoint_path) == source_checkpoint:
            return source_wrapper
        if Path(checkpoint_path) == evidence_checkpoint:
            raise AssertionError("Filterwise evidence checkpoint must not be loaded for pruning.")
        return SimpleNamespace(model=torch.load(checkpoint_path, weights_only=False))

    def prune(model, example_input, layer, indices):
        assert example_input.shape == (1, 3, 32, 32)
        if layer == "0":
            assert model is source_model
            assert indices == tuple(range(8))
            return after_first_layer
        assert layer == "1"
        assert model is after_first_layer
        assert indices == tuple(range(8))
        return final_model

    monkeypatch.setattr(compression_matrix, "_load_yolo_checkpoint", load_yolo)
    monkeypatch.setattr(compression_matrix, "run_filterwise_probe", prune)

    output = build_pruned_checkpoint(
        {
            "model": {"checkpoint": str(source_checkpoint)},
            "experiment": {"image_size": 32},
            "pruning": {
                "filterwise_manifest": str(manifest),
                "evidence_layer": "0",
                "evidence_filters_removed": 8,
                "candidate_layers": ["0", "1"],
                "cluster_size": 8,
                "prune_ratios": [0.25, 0.5],
                "allowed_map50_95_drop": 0.02,
                "importance": "minimum_weight",
            },
        },
        tmp_path / "output",
        requested_ratio=0.5,
    )

    assert output == tmp_path / "output" / "pruned.pt"
    assert output.exists()
    reloaded = torch.load(output, weights_only=False)
    assert [module.out_channels for module in reloaded] == [8, 8]
    record = json.loads((tmp_path / "output" / "pruning_summary.json").read_text())
    assert record["before"]["parameter_count"] == 336
    assert record["after"]["parameter_count"] == 104
    assert record["source_checkpoint"] == str(source_checkpoint)
    assert record["selected_ratio"] == 0.5


def test_build_pruned_checkpoint_skips_zero_cluster_layers_and_prunes_eligible_layers(
    monkeypatch, tmp_path
):
    dense_checkpoint = tmp_path / "dense.pt"
    dense_model = nn.Sequential(nn.Conv2d(3, 16, kernel_size=1), nn.Conv2d(16, 32, kernel_size=1))
    with torch.no_grad():
        dense_model[1].weight.copy_(
            torch.arange(32, dtype=torch.float32).reshape(32, 1, 1, 1).repeat(1, 16, 1, 1)
        )
    torch.save(dense_model, dense_checkpoint)
    evidence_checkpoint = tmp_path / "evidence.pt"
    torch.save(nn.Sequential(nn.Conv2d(3, 8, kernel_size=1)), evidence_checkpoint)
    manifest = tmp_path / "filterwise.json"
    manifest.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "layer": "0",
                        "filters_removed": 8,
                        "status": "screened_in",
                        "checkpoint_path": str(evidence_checkpoint),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    dense_wrapper = SimpleNamespace(model=dense_model)
    dense_wrapper.save = lambda path: torch.save(dense_wrapper.model, path)
    final_model = nn.Sequential(nn.Conv2d(3, 16, kernel_size=1), nn.Conv2d(16, 24, kernel_size=1))

    def load_yolo(checkpoint_path):
        if Path(checkpoint_path) == dense_checkpoint:
            return dense_wrapper
        if Path(checkpoint_path) == evidence_checkpoint:
            raise AssertionError("Filterwise evidence checkpoint must not be pruned.")
        return SimpleNamespace(model=torch.load(checkpoint_path, weights_only=False))

    def prune(model, example_input, layer, indices):
        assert model is dense_model
        assert example_input.shape == (1, 3, 32, 32)
        assert (layer, indices) == ("1", tuple(range(8)))
        return final_model

    monkeypatch.setattr(compression_matrix, "_load_yolo_checkpoint", load_yolo)
    monkeypatch.setattr(compression_matrix, "run_filterwise_probe", prune)

    build_pruned_checkpoint(
        {
            "model": {"checkpoint": str(dense_checkpoint)},
            "experiment": {"image_size": 32},
            "pruning": {
                "filterwise_manifest": str(manifest),
                "evidence_layer": "0",
                "evidence_filters_removed": 8,
                "candidate_layers": ["0", "1"],
                "cluster_size": 8,
                "prune_ratios": [0.3],
                "allowed_map50_95_drop": 0.02,
                "importance": "minimum_weight",
            },
        },
        tmp_path / "output",
    )

    record = json.loads((tmp_path / "output" / "pruning_summary.json").read_text())
    assert record["prune_indices"] == {"0": [], "1": list(range(8))}
    assert record["skipped_layers"] == ["0"]
