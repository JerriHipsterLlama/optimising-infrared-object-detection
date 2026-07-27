import csv
import json
from pathlib import Path

import pytest

from infrared_detection.evaluation.compression_matrix import (
    load_compression_config,
    planned_variants,
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
    assert pruning["pruning_ratio"] == 0.25
    assert planned_variants(config)[3]["provenance"] == {
        "filterwise_manifest": pruning["filterwise_manifest"],
        "candidate_layer": pruning["candidate_layer"],
        "cluster_size": pruning["cluster_size"],
        "pruning_ratio": pruning["pruning_ratio"],
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
