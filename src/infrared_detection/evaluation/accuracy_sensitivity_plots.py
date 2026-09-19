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
    map50: float | None
    map50_95: float
    precision: float | None = None
    recall: float | None = None


@dataclass(frozen=True)
class LayerCurve:
    name: str
    region: str
    compression_percent: tuple[float, ...]
    map50: tuple[float, ...] | None
    map50_95: tuple[float, ...]
    precision: tuple[float, ...] | None = None
    recall: tuple[float, ...] | None = None


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
    baselines = [row for row in rows if row.get("status") == "BASELINE" or row.get("candidate_id", "").lower() == "baseline"]
    if len(baselines) != 1:
        raise ValueError(f"Expected exactly one BASELINE row, found {len(baselines)}.")
    baseline = BaselineMetrics(
        map50=_number(baselines[0], "map50", required=False),
        map50_95=_number(baselines[0], "map50_95"),
        precision=_number(baselines[0], "precision", required=False),
        recall=_number(baselines[0], "recall", required=False),
    )

    grouped: dict[str, list[tuple[float, float, float, float | None, float | None, str]]] = {}
    for row in rows:
        if row.get("candidate_id", "").lower() == "baseline":
            continue
        if row.get("status") not in ("COMPLETED", None, ""):
            continue
        name = row.get("pruning_unit", "")
        if not name:
            raise ValueError("A COMPLETED row is missing pruning_unit.")
        grouped.setdefault(name, []).append(
            (
                _number(row, "actual_pruning_ratio"),
                _number(row, "map50", required=False),
                _number(row, "map50_95"),
                _number(row, "precision", required=False),
                _number(row, "recall", required=False),
                row.get("architectural_region", "unknown"),
            )
        )

    series: list[LayerCurve] = []
    for name in sorted(grouped):
        points = sorted(grouped[name], key=lambda item: item[0])
        if len({point[0] for point in points}) != len(points):
            raise ValueError(f"Duplicate compression ratios found for {name!r}.")
        region = points[0][5]
        if any(point[5] != region for point in points):
            raise ValueError(f"Multiple architectural regions found for {name!r}.")
        series.append(
            LayerCurve(
                name=name,
                region=region,
                compression_percent=(0.0, *tuple(point[0] * 100.0 for point in points)),
                map50=(
                    None
                    if baseline.map50 is None or any(point[1] is None for point in points)
                    else (baseline.map50, *tuple(float(point[1]) for point in points))
                ),
                map50_95=(baseline.map50_95, *tuple(point[2] for point in points)),
                precision=(
                    None
                    if baseline.precision is None or any(point[3] is None for point in points)
                    else (baseline.precision, *tuple(float(point[3]) for point in points))
                ),
                recall=(
                    None
                    if baseline.recall is None or any(point[4] is None for point in points)
                    else (baseline.recall, *tuple(float(point[4]) for point in points))
                ),
            )
        )
    if not series:
        raise ValueError("No COMPLETED pruning-layer rows found.")
    return PlotData(baseline=baseline, series=tuple(series))




def _layer_colors(series: tuple[LayerCurve, ...]) -> list[object]:
    """Choose categorical colors for small region-specific layer sets."""

    if len(series) <= 10:
        cmap = plt.get_cmap("tab10")
        return [cmap(index) for index in range(len(series))]
    if len(series) <= 20:
        cmap = plt.get_cmap("tab20")
        return [cmap(index) for index in range(len(series))]
    cmap = plt.get_cmap("turbo", len(series))
    return [cmap(index) for index in range(len(series))]

def _region_data(data: PlotData, region: str | None) -> PlotData:
    """Return a view containing all layers or one architectural region."""

    if region is None:
        return data
    series = tuple(curve for curve in data.series if curve.region == region)
    if not series:
        raise ValueError(f"No completed curves found for architectural region {region!r}.")
    return PlotData(baseline=data.baseline, series=series)



def _plot_accuracy_recall(
    data: PlotData,
    output: Path,
    *,
    region: str | None = None,
) -> Path | None:
    """Write paired mAP50-95/recall plots with shared layer colors."""

    if data.baseline.recall is None or any(curve.recall is None for curve in data.series):
        return None
    if region is not None and not any(curve.region == region for curve in data.series):
        return None
    scoped = _region_data(data, region)
    region_label = "all layers" if region is None else region.replace("_", " ")
    figure = plt.figure(figsize=(19.5, 11), dpi=180)
    grid = figure.add_gridspec(1, 2, width_ratios=(6.5, 1.5), wspace=0.03)
    axis = figure.add_subplot(grid[0, 0])
    auxiliary_axis = axis.twinx()
    legend_axis = figure.add_subplot(grid[0, 1])
    legend_axis.axis("off")
    colors = _layer_colors(scoped.series)
    for color, curve in zip(colors, scoped.series):
        axis.plot(
            curve.compression_percent,
            curve.map50_95,
            color=color,
            linewidth=1.35,
            marker="o",
            markersize=3.2,
            linestyle="-",
            alpha=0.9,
            label=curve.name,
        )
        auxiliary_axis.plot(
            curve.compression_percent,
            curve.recall,
            color=color,
            linewidth=1.35,
            marker="o",
            markersize=3.2,
            linestyle=":",
            alpha=0.9,
        )
    axis.set_title(f"mAP50-95 and recall by pruning layer ({region_label})", color="#202124", pad=14)
    axis.set_xlabel("Achieved channel pruning / compression (%)")
    axis.set_ylabel("mAP50-95 accuracy")
    auxiliary_axis.set_ylabel("Recall")
    max_compression = max(max(curve.compression_percent) for curve in scoped.series)
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
        ncol=1,
        title="Pruning layer",
        columnspacing=0.8,
        handlelength=1.6,
    )
    legend_axis.add_artist(layer_legend)
    metric_handles = [
        Line2D([0], [0], color="#202124", linestyle="-", label="mAP50-95"),
        Line2D([0], [0], color="#202124", linestyle=":", label="Recall"),
    ]
    legend_axis.legend(
        metric_handles,
        ["mAP50-95", "Recall"],
        loc="lower left",
        frameon=False,
        fontsize=8,
        title="Metric",
    )
    figure.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.10)
    filename = "map50_95_and_recall.png" if region is None else f"map50_95_and_recall_{region}.png"
    path = output / filename
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_accuracy_auxiliary(
    data: PlotData,
    output: Path,
    *,
    auxiliary_field: str,
    auxiliary_label: str,
    filename_stem: str,
    region: str,
) -> Path | None:
    """Write mAP50-95 against one auxiliary validation metric for a region."""

    if not any(curve.region == region for curve in data.series):
        return None
    if region is not None and not any(curve.region == region for curve in data.series):
        return None
    scoped = _region_data(data, region)
    if any(getattr(curve, auxiliary_field) is None for curve in scoped.series):
        return None
    figure = plt.figure(figsize=(19.5, 11), dpi=180)
    grid = figure.add_gridspec(1, 2, width_ratios=(6.5, 1.5), wspace=0.03)
    axis = figure.add_subplot(grid[0, 0])
    auxiliary_axis = axis.twinx()
    legend_axis = figure.add_subplot(grid[0, 1])
    legend_axis.axis("off")
    colors = _layer_colors(scoped.series)
    for color, curve in zip(colors, scoped.series):
        axis.plot(
            curve.compression_percent,
            curve.map50_95,
            color=color,
            linewidth=1.35,
            marker="o",
            markersize=3.2,
            linestyle="-",
            alpha=0.9,
            label=curve.name,
        )
        auxiliary_axis.plot(
            curve.compression_percent,
            getattr(curve, auxiliary_field),
            color=color,
            linewidth=1.35,
            marker="o",
            markersize=3.2,
            linestyle=":",
            alpha=0.9,
        )
    axis.set_title(
        f"mAP50-95 and {auxiliary_label.lower()} by pruning layer ({region})",
        color="#202124",
        pad=14,
    )
    axis.set_xlabel("Achieved channel pruning / compression (%)")
    axis.set_ylabel("mAP50-95 accuracy")
    auxiliary_axis.set_ylabel(auxiliary_label)
    max_compression = max(max(curve.compression_percent) for curve in scoped.series)
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
        ncol=1,
        title="Pruning layer",
        columnspacing=0.8,
        handlelength=1.6,
    )
    legend_axis.add_artist(layer_legend)
    metric_handles = [
        Line2D([0], [0], color="#202124", linestyle="-", label="mAP50-95"),
        Line2D([0], [0], color="#202124", linestyle=":", label=auxiliary_label),
    ]
    legend_axis.legend(
        metric_handles,
        ["mAP50-95", auxiliary_label],
        loc="lower left",
        frameon=False,
        fontsize=8,
        title="Metric",
    )
    figure.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.10)
    path = output / f"{filename_stem}_{region}.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path

def _plot_recall(
    data: PlotData,
    output: Path,
    *,
    region: str | None = None,
) -> Path | None:
    """Write a recall/compression plot, optionally limited to one region."""

    if data.baseline.recall is None or any(curve.recall is None for curve in data.series):
        return None
    if region is not None and not any(curve.region == region for curve in data.series):
        return None
    scoped = _region_data(data, region)
    region_label = "all layers" if region is None else region.replace("_", " ")
    figure = plt.figure(figsize=(19.5, 10), dpi=180)
    grid = figure.add_gridspec(1, 2, width_ratios=(6.5, 1.5), wspace=0.03)
    axis = figure.add_subplot(grid[0, 0])
    legend_axis = figure.add_subplot(grid[0, 1])
    legend_axis.axis("off")
    colors = _layer_colors(scoped.series)
    for color, curve in zip(colors, scoped.series):
        axis.plot(
            curve.compression_percent,
            curve.recall,
            color=color,
            linewidth=1.35,
            marker="o",
            markersize=3.2,
            linestyle="-",
            alpha=0.9,
            label=curve.name,
        )
    axis.set_title(f"Recall by pruning layer ({region_label})", color="#202124", pad=14)
    axis.set_xlabel("Achieved channel pruning / compression (%)")
    axis.set_ylabel("Recall")
    max_compression = max(max(curve.compression_percent) for curve in scoped.series)
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
        ncol=1,
        title="Pruning layer",
        columnspacing=0.8,
        handlelength=1.6,
    )
    figure.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.10)
    filename = "recall.png" if region is None else f"recall_{region}.png"
    path = output / filename
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path

def _plot_metric(
    data: PlotData,
    output: Path,
    *,
    field: str,
    ylabel: str,
    title: str,
) -> Path:
    figure = plt.figure(figsize=(19.5, 10), dpi=180)
    grid = figure.add_gridspec(1, 2, width_ratios=(6.5, 1.5), wspace=0.03)
    axis = figure.add_subplot(grid[0, 0])
    legend_axis = figure.add_subplot(grid[0, 1])
    legend_axis.axis("off")
    colors = _layer_colors(data.series)
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
        ncol=1,
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
        "map50_95": _plot_metric(
            data,
            output,
            field="map50_95",
            ylabel="mAP50-95",
            title="mAP50-95 by pruning layer",
        ),
    }
    if data.baseline.map50 is not None and all(curve.map50 is not None for curve in data.series):
        paths["map50"] = _plot_metric(
            data,
            output,
            field="map50",
            ylabel="mAP50",
            title="mAP50 by pruning layer",
        )
    for key, region in (
        ("recall", None),
        ("recall_neck", "neck"),
        ("recall_detect_head", "detect_head"),
    ):
        path = _plot_recall(data, output, region=region)
        if path is not None:
            paths[key] = path
    for key, region in (
        ("map50_95_and_recall", None),
        ("map50_95_and_recall_backbone", "backbone"),
        ("map50_95_and_recall_neck", "neck"),
        ("map50_95_and_recall_detect_head", "detect_head"),
    ):
        path = _plot_accuracy_recall(data, output, region=region)
        if path is not None:
            paths[key] = path
    for key, region in (
        ("map50_95_and_precision_backbone", "backbone"),
        ("map50_95_and_precision_neck", "neck"),
        ("map50_95_and_precision_detect_head", "detect_head"),
    ):
        path = _plot_accuracy_auxiliary(
            data,
            output,
            auxiliary_field="precision",
            auxiliary_label="Precision",
            filename_stem="map50_95_and_precision",
            region=region,
        )
        if path is not None:
            paths[key] = path

    if data.baseline.map50 is None or any(curve.map50 is None for curve in data.series):
        return paths

    figure = plt.figure(figsize=(19.5, 11), dpi=180)
    grid = figure.add_gridspec(1, 2, width_ratios=(6.5, 1.5), wspace=0.03)
    axis = figure.add_subplot(grid[0, 0])
    legend_axis = figure.add_subplot(grid[0, 1])
    legend_axis.axis("off")
    colors = _layer_colors(data.series)
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
        ncol=1,
        title="Pruning layer",
        columnspacing=0.8,
        handlelength=1.6,
    )
    legend_axis.add_artist(layer_legend)
    metric_handles = [
        Line2D([0], [0], color="#202124", linestyle=COMBINED_LINESTYLES["map50"], label="mAP50"),
        Line2D([0], [0], color="#202124", linestyle=COMBINED_LINESTYLES["map50_95"], label="mAP50-95"),
    ]
    legend_axis.legend(
        metric_handles,
        ["mAP50", "mAP50-95"],
        loc="lower left",
        frameon=False,
        fontsize=8,
        title="Metric",
    )
    figure.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.10)
    combined = output / "map50_and_map50_95.png"
    figure.savefig(combined, bbox_inches="tight")
    plt.close(figure)
    paths["combined"] = combined
    return paths



def show_interactive_gallery(paths: dict[str, Path]) -> None:
    """Open saved figures in interactive Matplotlib windows."""

    for name, path in paths.items():
        image = plt.imread(path)
        figure, axis = plt.subplots(figsize=(16, 9), dpi=120)
        axis.imshow(image)
        axis.axis("off")
        figure.canvas.manager.set_window_title(name)
        figure.tight_layout(pad=0)
    plt.show()

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot YOLOv8n accuracy-sensitivity results.")
    parser.add_argument("--results", required=True, help="Path to screening results.csv")
    parser.add_argument("--output-dir", help="Directory for PNGs; defaults beside results.csv")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open the generated PNGs in interactive Matplotlib windows.",
    )
    args = parser.parse_args()
    output = args.output_dir or str(Path(args.results).resolve().parent / "plots")
    if args.interactive:
        try:
            plt.switch_backend("TkAgg")
        except ImportError as error:
            raise RuntimeError(
                "Interactive plotting requires a Tk Matplotlib backend. "
                "Run without --interactive to export PNGs."
            ) from error
    paths = plot_accuracy_curves(args.results, output)
    for name, path in paths.items():
        print(f"{name}: {path}")
    if args.interactive:
        show_interactive_gallery(paths)


if __name__ == "__main__":
    main()
