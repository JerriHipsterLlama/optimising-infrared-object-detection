from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


def _load_report_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "tools" / "generate_supervisor_pruning_report.py"
    spec = importlib.util.spec_from_file_location("generate_supervisor_pruning_report", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_results(path: Path, rows: list[dict[str, object]]) -> Path:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_report_main_uses_completed_single_layer_curves_from_multiple_inputs(monkeypatch, tmp_path):
    report = _load_report_module()
    nano = _write_results(
        tmp_path / "nano.csv",
        [
            {"candidate_id": "baseline", "stage": "baseline", "status": "completed", "model_variant": "yolov8n", "hardware": "rtx3070", "map50_95": 0.75},
            {"candidate_id": "n-1", "stage": "single_layer", "status": "completed", "model_variant": "yolov8n", "hardware": "rtx3070", "layer": "model.6.m.0.cv1.conv", "filters_remaining": 63, "map50_95": 0.74, "latency_p50_ms": 3.2},
            {"candidate_id": "n-2", "stage": "single_layer", "status": "failed", "model_variant": "yolov8n", "hardware": "rtx3070", "layer": "model.6.m.0.cv1.conv", "filters_remaining": 62, "map50_95": 0.10, "latency_p50_ms": 1.0},
        ],
    )
    medium = _write_results(
        tmp_path / "medium.csv",
        [
            {"candidate_id": "m-1", "stage": "single_layer", "status": "completed", "model_variant": "yolov8m", "hardware": "jetson_orin_nano", "layer": "model.8.m.0.cv2.conv", "filters_remaining": 191, "map50_95": 0.79, "latency_p50_ms": 8.4},
            {"candidate_id": "m-2", "stage": "single_layer", "status": "planned", "model_variant": "yolov8m", "hardware": "jetson_orin_nano", "layer": "model.8.m.0.cv2.conv", "filters_remaining": 190},
        ],
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(report, "_load_manifest", lambda name: [])
    monkeypatch.setattr(report, "plot_backbone", lambda run_1, run_2, output: [])
    monkeypatch.setattr(report, "plot_neck", lambda rows, output: [])
    monkeypatch.setattr(
        report,
        "write_briefing",
        lambda output, single_layer, backbone, neck: captured.update(single_layer=single_layer),
    )

    output = tmp_path / "report"
    report.main(
        [
            "--output", str(output),
            "--single-layer-results", str(nano),
            "--single-layer-results", str(medium),
        ]
    )

    grouped = captured["single_layer"]
    assert set(grouped) == {
        "yolov8m/jetson_orin_nano/model.8.m.0.cv2.conv",
        "yolov8n/rtx3070/model.6.m.0.cv1.conv",
    }
    assert [row["candidate_id"] for rows in grouped.values() for row in rows] == ["m-1", "n-1"]
    assert {(int(row["filters_remaining"]), float(row["map50_95"]), float(row["latency_p50_ms"])) for rows in grouped.values() for row in rows} == {
        (63, 0.74, 3.2),
        (191, 0.79, 8.4),
    }
    assert (output / "01_single_layer_accuracy.png").exists()
    assert (output / "02_single_layer_latency.png").exists()
