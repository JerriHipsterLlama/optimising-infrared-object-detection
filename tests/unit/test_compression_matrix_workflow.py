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


def test_planned_variants_include_dense_and_pruned_precision_matrix():
    rows = planned_variants(
        {
            "pruning": {
                "enabled": True,
                "candidate_id": "cluster-8-ratio-0.25",
                "filterwise_manifest": "artifacts/filterwise_rtx_screening/manifest.json",
                "candidate_layer": "model.8.cv2.conv",
                "cluster_size": 8,
                "pruning_ratio": 0.25,
            },
            "precisions": ["fp32", "fp16", "int8"],
        }
    )

    assert [row["variant_id"] for row in rows] == [
        "dense-fp32",
        "dense-fp16",
        "dense-int8",
        "cluster-8-ratio-0.25-fp32",
        "cluster-8-ratio-0.25-fp16",
        "cluster-8-ratio-0.25-int8",
    ]
    assert all(row["status"] == "planned" for row in rows)
    assert all("provenance" in row for row in rows)
    pruned = rows[3]
    assert pruned["provenance"] == {
        "filterwise_manifest": "artifacts/filterwise_rtx_screening/manifest.json",
        "candidate_layer": "model.8.cv2.conv",
        "cluster_size": 8,
        "pruning_ratio": 0.25,
    }


def test_planned_variants_reject_non_official_precision():
    with pytest.raises(ValueError, match="Official precisions"):
        planned_variants({"precisions": ["fp32", "int4"]})


@pytest.mark.parametrize(
    "precisions",
    [["fp32", "fp16"], ["fp32", "fp16", "int8", "int8"], ["int8", "fp16", "fp32"]],
)
def test_precision_matrix_requires_exact_ordered_official_set(precisions):
    with pytest.raises(ValueError, match="exactly.*ordered"):
        planned_variants({"precisions": precisions})


def test_load_compression_config_reads_yaml(tmp_path):
    config_path = tmp_path / "compression.yaml"
    config_path.write_text(
        "pruning:\n  enabled: false\nprecisions: [fp32, fp16, int8]\n",
        encoding="utf-8",
    )

    config = load_compression_config(config_path)

    assert config["precisions"] == ["fp32", "fp16", "int8"]
    assert config["_config_path"] == str(config_path.resolve())


def test_load_compression_config_rejects_incomplete_precision_matrix(tmp_path):
    config_path = tmp_path / "compression.yaml"
    config_path.write_text("precisions: [fp32, fp16]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly.*ordered"):
        load_compression_config(config_path)


def test_rtx_config_declares_filterwise_candidate_without_inference():
    config_path = Path("configs/experiments/rtx_compression_matrix.yaml")

    config = load_compression_config(config_path)
    pruning = config["pruning"]

    assert pruning["filterwise_manifest"] == "artifacts/filterwise_rtx_screening/manifest.json"
    assert pruning["candidate_layer"] == "model.8.cv2.conv"
    assert pruning["cluster_size"] == 8
    assert pruning["filter_removal_count"] == 8
    assert planned_variants(config)[3]["provenance"] == {
        "filterwise_manifest": pruning["filterwise_manifest"],
        "candidate_layer": pruning["candidate_layer"],
        "cluster_size": pruning["cluster_size"],
        "filter_removal_count": pruning["filter_removal_count"],
    }


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


def test_build_pruned_checkpoint_removes_selected_filters_and_records_reloaded_metrics(
    monkeypatch, tmp_path
):
    source_checkpoint = tmp_path / "candidate.pt"
    source_model = nn.Sequential(nn.Conv2d(3, 16, kernel_size=1))
    with torch.no_grad():
        source_model[0].weight.copy_(
            torch.arange(16, dtype=torch.float32).reshape(16, 1, 1, 1).repeat(1, 3, 1, 1)
        )
    torch.save(source_model, source_checkpoint)
    manifest = tmp_path / "filterwise.json"
    manifest.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "layer": "0",
                        "filters_removed": 8,
                        "status": "screened_in",
                        "checkpoint_path": str(source_checkpoint),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    source_wrapper = SimpleNamespace(model=source_model)
    source_wrapper.save = lambda path: torch.save(source_wrapper.model, path)
    pruned_model = nn.Sequential(nn.Conv2d(3, 8, kernel_size=1))

    def load_yolo(checkpoint_path):
        if Path(checkpoint_path) == source_checkpoint:
            return source_wrapper
        return SimpleNamespace(model=torch.load(checkpoint_path, weights_only=False))

    def prune(model, example_input, layer, indices):
        assert model is source_model
        assert example_input.shape == (1, 3, 32, 32)
        assert (layer, indices) == ("0", tuple(range(8)))
        return pruned_model

    monkeypatch.setattr(compression_matrix, "_load_yolo_checkpoint", load_yolo)
    monkeypatch.setattr(compression_matrix, "run_filterwise_probe", prune)

    output = build_pruned_checkpoint(
        {
            "model": {"checkpoint": str(source_checkpoint)},
            "experiment": {"image_size": 32},
            "pruning": {
                "filterwise_manifest": str(manifest),
                "candidate_layer": "0",
                "filter_removal_count": 8,
            },
        },
        tmp_path / "output",
    )

    assert output == tmp_path / "output" / "pruned.pt"
    assert output.exists()
    assert torch.load(output, weights_only=False)[0].out_channels == 8
    record = json.loads((tmp_path / "output" / "pruning_summary.json").read_text())
    assert record["before"]["parameter_count"] == 64
    assert record["after"]["parameter_count"] == 32
    assert record["source_checkpoint"] == str(source_checkpoint)
