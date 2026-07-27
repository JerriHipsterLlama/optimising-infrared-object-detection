import csv
import json

import pytest

from infrared_detection.evaluation.compression_matrix import (
    load_compression_config,
    planned_variants,
    write_compression_manifest,
)


def test_planned_variants_include_dense_and_pruned_precision_matrix():
    rows = planned_variants(
        {
            "pruning": {"enabled": True, "candidate_id": "cluster-8-ratio-0.25"},
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


def test_planned_variants_reject_non_official_precision():
    with pytest.raises(ValueError, match="Official precisions"):
        planned_variants({"precisions": ["fp32", "int4"]})


def test_load_compression_config_reads_yaml_and_rejects_non_official_precision(tmp_path):
    config_path = tmp_path / "compression.yaml"
    config_path.write_text(
        "pruning:\n  enabled: false\nprecisions: [fp32, fp16, int8]\n",
        encoding="utf-8",
    )

    config = load_compression_config(config_path)

    assert config["precisions"] == ["fp32", "fp16", "int8"]
    assert config["_config_path"] == str(config_path.resolve())


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
