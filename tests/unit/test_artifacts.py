from __future__ import annotations

import csv
import json

from infrared_detection.evaluation.artifacts import write_metrics_csv


def test_metrics_csv_places_research_columns_first_and_diagnostics_last(tmp_path):
    output = tmp_path / "results.csv"
    write_metrics_csv(
        output,
        [{"error": "boom", "map50_95": 0.8, "candidate_id": "c1", "status": "failed", "zeta": 4, "reason": "x"}],
        leading_columns=("candidate_id", "status", "map50_95"),
        trailing_columns=("reason", "error"),
    )

    header = output.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == ["candidate_id", "status", "map50_95", "zeta", "reason", "error"]


def test_metrics_csv_serializes_nested_values_and_keeps_one_physical_row(tmp_path):
    output = tmp_path / "results.csv"

    write_metrics_csv(
        output,
        [
            {
                "variant_id": "dense-fp32",
                "commands": {"build": ["trtexec", "--onnx=model.onnx"]},
                "stderr": "line one\nline two",
            }
        ],
    )

    content = output.read_text(encoding="utf-8")
    assert content.count("\n") == 2
    with output.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert json.loads(row["commands"]) == {"build": ["trtexec", "--onnx=model.onnx"]}
    assert row["stderr"] == "line one\\nline two"
