"""Matplotlib plots for YOLOv8n accuracy-sensitivity screening results."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

COMBINED_LINESTYLES = {"map50": ":", "map50_95": "-"}
SINGLE_LINESTYLES = {"map50": "-", "map50_95": "-"}
LEGEND_COLUMNS = 4
X_AXIS_RIGHT_PADDING_PERCENT = 1.0


@dataclass(frozen=True)
class BaselineMetrics:
    map50: float
    map50_95: float


@dataclass(frozen=True)
class LayerCurve:
    name: str
    region: str
    compression_percent: tuple[float, ...]
    map50: tuple[float, ...]
    map50_95: tuple[float, ...]


@dataclass(frozen=True)
class PlotData:
    baseline: BaselineMetrics
    series: tuple[LayerCurve, ...]


def _number(row: dict[str, str], field: str, *, required: bool = True) -> float | None:
    value = row.get(field, "")
    if value in (None, ""):
        if required:
            raise ValueError(f"Results row is missing {field!r}.")
        return None
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"Results field {field!r} must be finite.")
    return number


def load_plot_data(results_csv: str | Path) -> PlotData:
    """Load baseline and successful layer curves from a screening CSV."""

    path = Path(results_csv)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    baselines = [row for row in rows if row.get("status") == "BASELINE"]
    if len(baselines) != 1:
        raise ValueError(f"Expected exactly one BASELINE row, found {len(baselines)}.")
    baseline = BaselineMetrics(
        map50=_number(baselines[0], "map50"),
        map50_95=_number(baselines[0], "map50_95"),
    )

    grouped: dict[str, list[tuple[float, float, float, str]]] = {}
    for row in rows:
        if row.get("status") != "COMPLETED":
            continue
        name = row.get("pruning_unit", "")
        if not name:
            raise ValueError("A COMPLETED row is missing pruning_unit.")
        grouped.setdefault(name, []).append(
            (
                _number(row, "actual_pruning_ratio"),
                _number(row, "map50"),
                _number(row, "map50_95"),
                row.get("architectural_region", "unknown"),
            )
        )

    series: list[LayerCurve] = []
    for name in sorted(grouped):
        points = sorted(grouped[name], key=lambda item: item[0])
        if len({point[0] for point in points}) != len(points):
            raise ValueError(f"Duplicate compression ratios found for {name!r}.")
        region = points[0][3]
        if any(point[3] != region for point in points):
            raise ValueError(f"Multiple architectural regions found for {name!r}.")
        series.append(
            LayerCurve(
                name=name,
                region=region,
                compression_percent=(0.0, *tuple(point[0] * 100.0 for point in points)),
                map50=(baseline.map50, *tuple(point[1] for point in points)),
                map50_95=(baseline.map50_95, *tuple(point[2] for point in points)),
            )
        )
    if not series:
        raise ValueError("No COMPLETED pruning-layer rows found.")
    return PlotData(baseline=baseline, series=tuple(series))


def _plot_metric(
    data: PlotData,
    output: Path,
    *,
    field: str,
    ylabel: str,
    title: str,
) -> Path:
    figure = plt.figure(figsize=(24, 10), dpi=180)
    grid = figure.add_gridspec(1, 2, width_ratios=(5.0, 2.2), wspace=0.04)
    axis = figure.add_subplot(grid[0, 0])
    legend_axis = figure.add_subplot(grid[0, 1])
    legend_axis.axis("off")
    colors = plt.colormaps["viridis"](np.linspace(0.05, 0.95, len(data.series)))
    for color, curve in zip(colors, data.series):
        axis.plot(
            curve.compression_percent,
            getattr(curve, field),
            color=color,
            linewidth=1.35,
            marker="o",
            markersize=3.2,
            alpha=0.9,
            linestyle=SINGLE_LINESTYLES[field],
            label=curve.name,
        )
    axis.set_title(title, color="#202124", pad=14)
    axis.set_xlabel("Achieved channel pruning / compression (%)")
    axis.set_ylabel(ylabel)
    max_compression = max(max(curve.compression_percent) for curve in data.series)
    axis.set_xlim(0.0, max_compression + X_AXIS_RIGHT_PADDING_PERCENT)
    axis.grid(True, color="#d9dde3", linewidth=0.7, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    handles, labels = axis.get_legend_handles_labels()
    legend_axis.legend(
        handles,
        labels,
        loc="upper left",
        frameon=False,
        fontsize=8,
        ncol=2,
        title="Pruning layer",
        columnspacing=0.8,
        handlelength=1.6,
    )
    figure.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.10)
    path = output / ("map50.png" if field == "map50" else "map50_95.png")
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_accuracy_curves(results_csv: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Write mAP50, mAP50-95, and combined layer-overlay plots."""

    data = load_plot_data(results_csv)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "map50": _plot_metric(
            data,
            output,
            field="map50",
            ylabel="mAP50",
            title="mAP50 by pruning layer",
        ),
        "map50_95": _plot_metric(
            data,
            output,
            field="map50_95",
            ylabel="mAP50-95",
            title="mAP50-95 by pruning layer",
        ),
    }

    figure = plt.figure(figsize=(24, 11), dpi=180)
    grid = figure.add_gridspec(1, 2, width_ratios=(5.0, 2.2), wspace=0.04)
    axis = figure.add_subplot(grid[0, 0])
    legend_axis = figure.add_subplot(grid[0, 1])
    legend_axis.axis("off")
    colors = plt.colormaps["viridis"](np.linspace(0.05, 0.95, len(data.series)))
    for color, curve in zip(colors, data.series):
        label = curve.name
        axis.plot(
            curve.compression_percent,
            curve.map50,
            color=color,
            linewidth=1.3,
            marker="o",
            markersize=3.0,
            linestyle=COMBINED_LINESTYLES["map50"],
            label=label,
        )
        axis.plot(
            curve.compression_percent,
            curve.map50_95,
            color=color,
            linewidth=1.3,
            marker="o",
            markersize=3.0,
            linestyle=COMBINED_LINESTYLES["map50_95"],
        )
    axis.set_title("mAP50 and mAP50-95 by pruning layer", color="#202124", pad=14)
    axis.set_xlabel("Achieved channel pruning / compression (%)")
    axis.set_ylabel("Accuracy")
    max_compression = max(max(curve.compression_percent) for curve in data.series)
    axis.set_xlim(0.0, max_compression + X_AXIS_RIGHT_PADDING_PERCENT)
    axis.grid(True, color="#d9dde3", linewidth=0.7, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    handles, labels = axis.get_legend_handles_labels()
    layer_legend = legend_axis.legend(
        handles,
        labels,
        loc="upper left",
        frameon=False,
        fontsize=8,
        ncol=2,
        title="Pruning layer",
        columnspacing=0.8,
        handlelength=1.6,
    )
    metric_handles = [
        Line2D([0], [0], color="#202124", linestyle=COMBINED_LINESTYLES["map50"], label="mAP50"),
        Line2D([0], [0], color="#202124", linestyle=COMBINED_LINESTYLES["map50_95"], label="mAP50-95"),
    ]
    axis.legend(metric_handles, ["mAP50", "mAP50-95"], loc="upper right", frameon=False, fontsize=8, title="Metric")
    figure.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.10)
    combined = output / "map50_and_map50_95.png"
    figure.savefig(combined, bbox_inches="tight")
    plt.close(figure)
    paths["combined"] = combined
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot YOLOv8n accuracy-sensitivity results.")
    parser.add_argument("--results", required=True, help="Path to screening results.csv")
    parser.add_argument("--output-dir", help="Directory for PNGs; defaults beside results.csv")
    args = parser.parse_args()
    output = args.output_dir or str(Path(args.results).resolve().parent / "plots")
    for name, path in plot_accuracy_curves(args.results, output).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
