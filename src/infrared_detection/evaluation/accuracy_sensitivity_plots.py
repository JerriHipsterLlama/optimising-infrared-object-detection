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
import numpy as np


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
    linestyle: str = "-",
) -> Path:
    figure, axis = plt.subplots(figsize=(16, 10), dpi=180)
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
            linestyle=linestyle,
            label=f"{curve.name} ({curve.region})",
        )
    axis.set_title(title, color="#202124", pad=14)
    axis.set_xlabel("Achieved channel pruning / compression (%)")
    axis.set_ylabel(ylabel)
    axis.set_xlim(left=0.0)
    axis.grid(True, color="#d9dde3", linewidth=0.7, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=7,
        ncol=2,
        title="Pruning layer",
    )
    figure.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
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
            linestyle=":",
        ),
    }

    figure, axis = plt.subplots(figsize=(16, 10), dpi=180)
    colors = plt.colormaps["viridis"](np.linspace(0.05, 0.95, len(data.series)))
    for color, curve in zip(colors, data.series):
        label = f"{curve.name} ({curve.region})"
        axis.plot(curve.compression_percent, curve.map50, color=color, linewidth=1.3, marker="o", markersize=3.0, label=label)
        axis.plot(curve.compression_percent, curve.map50_95, color=color, linewidth=1.3, marker="o", markersize=3.0, linestyle=":")
    axis.set_title("mAP50 and mAP50-95 by pruning layer", color="#202124", pad=14)
    axis.set_xlabel("Achieved channel pruning / compression (%)")
    axis.set_ylabel("Accuracy")
    axis.set_xlim(left=0.0)
    axis.grid(True, color="#d9dde3", linewidth=0.7, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    layer_legend = axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=7, ncol=2, title="Pruning layer")
    axis.add_artist(layer_legend)
    axis.plot([], [], color="#202124", linestyle="-", label="mAP50")
    axis.plot([], [], color="#202124", linestyle=":", label="mAP50-95")
    axis.legend(loc="lower left", bbox_to_anchor=(1.01, 0.0), frameon=False, fontsize=8, title="Metric")
    figure.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
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
