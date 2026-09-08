from __future__ import annotations

import csv
from pathlib import Path

from infrared_detection.evaluation.accuracy_sensitivity_plots import (
    load_plot_data,
    plot_accuracy_curves,
)


def _write_results(path: Path) -> None:
    fields = [
        "status",
        "pruning_unit",
        "architectural_region",
        "actual_pruning_ratio",
        "map50_95",
        "map50",
    ]
    rows = [
        {"status": "BASELINE", "pruning_unit": "", "architectural_region": "", "actual_pruning_ratio": "", "map50_95": "0.70", "map50": "0.90"},
        {"status": "COMPLETED", "pruning_unit": "layer.b", "architectural_region": "neck", "actual_pruning_ratio": "0.25", "map50_95": "0.60", "map50": "0.82"},
        {"status": "COMPLETED", "pruning_unit": "layer.a", "architectural_region": "backbone", "actual_pruning_ratio": "0.125", "map50_95": "0.65", "map50": "0.86"},
        {"status": "GROUPED", "pruning_unit": "layer.skip", "architectural_region": "backbone", "actual_pruning_ratio": "", "map50_95": "", "map50": ""},
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_load_plot_data_filters_completed_rows_and_adds_baseline(tmp_path: Path) -> None:
    results = tmp_path / "results.csv"
    _write_results(results)

    data = load_plot_data(results)

    assert data.baseline.map50 == 0.90
    assert [series.name for series in data.series] == ["layer.a", "layer.b"]
    assert data.series[0].compression_percent == (0.0, 12.5)
    assert data.series[0].map50 == (0.90, 0.86)


def test_plot_accuracy_curves_writes_three_pngs(tmp_path: Path) -> None:
    results = tmp_path / "results.csv"
    output = tmp_path / "plots"
    _write_results(results)

    paths = plot_accuracy_curves(results, output)

    assert set(paths) == {"map50", "map50_95", "combined"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())
