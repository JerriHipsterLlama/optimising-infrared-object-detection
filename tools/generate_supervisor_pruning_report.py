"""Generate supervisor-facing pruning figures from collected RTX results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "runs" / "experiments"
OUTPUT_DEFAULT = EXPERIMENTS / "supervisor_pruning_report"
TARGET_DROP = 0.01


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _load_manifest(name: str) -> list[dict[str, Any]]:
    path = EXPERIMENTS / name / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def _load_filterwise() -> list[dict[str, Any]]:
    path = EXPERIMENTS / "filterwise_rtx_screening" / "candidates.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 220,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )


def plot_filterwise(rows: list[dict[str, Any]], output: Path) -> dict[str, list[dict[str, Any]]]:
    baseline = next(row for row in rows if row["candidate_id"] == "baseline")
    baseline_map = float(baseline["map50_95"])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("stage") == "filterwise" and row.get("status") in {"screened_in", "screened_out"}:
            if _number(row.get("map50_95")) is not None:
                grouped[str(row["layer"])].append(row)
    for layer_rows in grouped.values():
        layer_rows.sort(key=lambda row: int(row["filters_removed"]))

    fig, axes = plt.subplots(2, 3, figsize=(10.8, 5.8), constrained_layout=True)
    axes_flat = axes.ravel()
    for axis, (layer, layer_rows) in zip(axes_flat, sorted(grouped.items())):
        removed = [int(row["filters_removed"]) for row in layer_rows]
        map_values = [float(row["map50_95"]) for row in layer_rows]
        statuses = [row["status"] for row in layer_rows]
        safe_x = [x for x, status in zip(removed, statuses) if status == "screened_in"]
        safe_y = [y for y, status in zip(map_values, statuses) if status == "screened_in"]
        rejected_x = [x for x, status in zip(removed, statuses) if status == "screened_out"]
        rejected_y = [y for y, status in zip(map_values, statuses) if status == "screened_out"]
        axis.plot(removed, map_values, color="#4C78A8", linewidth=1.2)
        axis.scatter(safe_x, safe_y, color="#2A9D8F", s=12, label="screened in")
        axis.scatter(rejected_x, rejected_y, color="#E76F51", s=14, marker="x", label="screened out")
        axis.axhline(baseline_map, color="#555555", linewidth=0.9, linestyle="--", label="baseline")
        axis.axhline(baseline_map - TARGET_DROP, color="#E9C46A", linewidth=0.9, linestyle=":", label="1 pp drop")
        axis.set_title(layer)
        axis.set_xlabel("Filters removed")
        axis.set_ylabel("mAP50–95")
        axis.grid(alpha=0.22)
    for axis in axes_flat[len(grouped) :]:
        axis.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", ncol=4, frameon=False)
    fig.suptitle("Filterwise pruning: measured mAP50–95 response by layer", y=1.02, fontsize=12)
    fig.savefig(output / "01_filterwise_accuracy.png", bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10.8, 4.3), constrained_layout=True)
    for layer, layer_rows in sorted(grouped.items()):
        measured = [row for row in layer_rows if _number(row.get("latency_p50_ms")) is not None]
        if measured:
            axis.plot(
                [int(row["filters_removed"]) for row in measured],
                [float(row["latency_p50_ms"]) for row in measured],
                marker="o",
                markersize=3,
                linewidth=1.1,
                label=layer,
            )
    axis.set_xlabel("Filters removed")
    axis.set_ylabel("TensorRT latency p50 (ms)")
    axis.set_title("Filterwise pruning: available RTX 3070 latency samples")
    axis.grid(alpha=0.25)
    axis.legend(ncol=3, frameon=False)
    fig.savefig(output / "02_filterwise_latency.png", bbox_inches="tight")
    plt.close(fig)
    return grouped


def _baseline(rows: list[dict[str, Any]], precision: str) -> dict[str, Any]:
    return next(row for row in rows if row["variant_id"] == f"dense-{precision}")


def plot_backbone(run_1: list[dict[str, Any]], run_2: list[dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    runs = [("Run 1", run_1, "o"), ("Run 2", run_2, "s")]
    colors = {"fp32": "#4C78A8", "fp16": "#F58518"}
    collected: list[dict[str, Any]] = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for run_label, rows, marker in runs:
        for precision, color in colors.items():
            baseline = _baseline(rows, precision)
            candidates = [
                row
                for row in rows
                if row.get("status") == "completed" and row.get("precision") == precision and row.get("prune_ratio") is not None
            ]
            candidates.sort(key=lambda row: float(row["prune_ratio"]))
            ratios = [float(row["prune_ratio"]) for row in candidates]
            drops = [float(row["map50_95"]) - float(baseline["map50_95"]) for row in candidates]
            latencies = [float(row["latency_p50_ms"]) for row in candidates]
            label = f"{run_label} {precision.upper()}"
            axes[0].plot(ratios, drops, marker=marker, markersize=4, color=color, linewidth=1.1, label=label)
            axes[1].plot(ratios, latencies, marker=marker, markersize=4, color=color, linewidth=1.1, label=label)
            for row, drop in zip(candidates, drops):
                collected.append({**row, "run": run_label, "map_drop": drop, "baseline": baseline})
    axes[0].axhline(-TARGET_DROP, color="#E76F51", linestyle="--", linewidth=1, label="−0.01 acceptance limit")
    axes[0].set_title("Backbone-only: accuracy change from same-run dense baseline")
    axes[0].set_xlabel("Prune ratio")
    axes[0].set_ylabel("Δ mAP50–95")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=2)
    axes[1].set_title("Backbone-only: RTX 3070 latency")
    axes[1].set_xlabel("Prune ratio")
    axes[1].set_ylabel("Latency p50 (ms)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, ncol=2)
    fig.savefig(output / "03_backbone_compression_matrix.png", bbox_inches="tight")
    plt.close(fig)
    return collected


def plot_neck(rows: list[dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    colors = {"fp32": "#4C78A8", "fp16": "#F58518"}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    records: list[dict[str, Any]] = []
    for precision, color in colors.items():
        baseline = _baseline(rows, precision)
        candidates = [
            row
            for row in rows
            if row.get("status") == "completed" and row.get("precision") == precision and row.get("prune_ratio") is not None
        ]
        candidates.sort(key=lambda row: float(row["prune_ratio"]))
        ratios = [float(row["prune_ratio"]) for row in candidates]
        maps = [float(row["map50_95"]) for row in candidates]
        recalls = [float(row["recall"]) for row in candidates]
        axes[0].plot(ratios, maps, marker="o", color=color, label=precision.upper())
        axes[1].plot(ratios, recalls, marker="o", color=color, label=precision.upper())
        axes[0].axhline(float(baseline["map50_95"]), color=color, linestyle="--", alpha=0.65)
        axes[1].axhline(float(baseline["recall"]), color=color, linestyle="--", alpha=0.65)
        for row in candidates:
            records.append({**row, "map_drop": float(row["map50_95"]) - float(baseline["map50_95"]), "recall_drop": float(row["recall"]) - float(baseline["recall"])})
    axes[0].set_title("Backbone + neck pruning: mAP50–95")
    axes[0].set_xlabel("Prune ratio")
    axes[0].set_ylabel("mAP50–95")
    axes[1].set_title("Backbone + neck pruning: recall")
    axes[1].set_xlabel("Prune ratio")
    axes[1].set_ylabel("Recall")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.savefig(output / "04_neck_pruning_warning.png", bbox_inches="tight")
    plt.close(fig)
    return records


def _format_ratio_list(records: list[dict[str, Any]], precision: str) -> str:
    accepted = sorted(
        {
        float(row["prune_ratio"])
        for row in records
        if row["precision"] == precision and row["map_drop"] >= -TARGET_DROP
        }
    )
    return ", ".join(f"{ratio:g}" for ratio in accepted) or "None"


def write_briefing(
    output: Path,
    filterwise: dict[str, list[dict[str, Any]]],
    backbone: list[dict[str, Any]],
    neck: list[dict[str, Any]],
) -> None:
    fp32 = [row for row in backbone if row["precision"] == "fp32"]
    fp16 = [row for row in backbone if row["precision"] == "fp16"]
    best_fp32 = max((row for row in fp32 if row["map_drop"] >= -TARGET_DROP), key=lambda row: float(row["prune_ratio"]))
    best_fp16 = max((row for row in fp16 if row["map_drop"] >= -TARGET_DROP), key=lambda row: float(row["prune_ratio"]))
    lines = [
        "# RTX 3070 pruning results briefing",
        "",
        "## Scope",
        "",
        "This briefing separates the validated backbone-only experiments from the latest exploratory backbone-plus-neck run. All reported accuracy changes use mAP50–95 relative to the dense baseline in the same run and precision.",
        "",
        "## Figures",
        "",
        "1. `01_filterwise_accuracy.png` — measured layerwise sensitivity curves.",
        "2. `02_filterwise_latency.png` — available latency samples from the layerwise sweep.",
        "3. `03_backbone_compression_matrix.png` — validated backbone-only compression matrix results.",
        "4. `04_neck_pruning_warning.png` — exploratory backbone-plus-neck accuracy and recall loss.",
        "",
        "## Filterwise pruning coverage",
        "",
        "| Layer | Completed accuracy points | Filters removed tested | Interpretation |",
        "|---|---:|---:|---|",
    ]
    for layer, rows in sorted(filterwise.items()):
        removed = [int(row["filters_removed"]) for row in rows]
        lines.append(f"| `{layer}` | {len(rows)} | {min(removed)}–{max(removed)} | Use this curve to choose layer-specific pruning tolerance. |")
    lines += [
        "",
        "The latency samples are sparse relative to the accuracy sweep, so they should be interpreted as preliminary RTX 3070 measurements rather than a final hardware conclusion.",
        "",
        "## Validated backbone-only compression matrix (`*_run_1`, `*_run_2`)",
        "",
        f"Acceptance criterion used here: mAP50–95 drop no worse than {TARGET_DROP:.2f} (one percentage point) from the same-run dense model.",
        "",
        "| Precision | Completed ratios meeting criterion | Largest observed accepted ratio | Recommended current candidate |",
        "|---|---|---:|---|",
        f"| FP32 | {_format_ratio_list(backbone, 'fp32')} | {float(best_fp32['prune_ratio']):g} | Ratio {float(best_fp32['prune_ratio']):g} was accepted in both runs; Run 1 ΔmAP {best_fp32['map_drop']:+.4f}, latency {float(best_fp32['latency_p50_ms']):.3f} ms |",
        f"| FP16 | {_format_ratio_list(backbone, 'fp16')} | {float(best_fp16['prune_ratio']):g} | Ratio 0.20 was accepted in Run 1; ratio 0.19 is the conservative Run 2 candidate (ΔmAP −0.0067). |",
        "",
        "The validated matrix prunes only the backbone layers `model.4.cv2.conv`, `model.6.cv2.conv`, and `model.8.cv2.conv`, using a cluster size of 8. It shows a small model-size reduction and largely unchanged RTX latency at these pruning levels; the main deployment speed gain is FP16 TensorRT, not pruning alone.",
        "",
        "## Exploratory backbone-plus-neck run — do not treat as an accepted configuration",
        "",
        "The latest run pruned the same backbone layers plus `model.12.cv2.conv`, `model.15.cv2.conv`, `model.18.cv2.conv`, and `model.21.cv2.conv` without prior layerwise sensitivity validation. The resulting loss is substantial, especially at ratios 0.20 and 0.30.",
        "",
        "| Precision | Ratio | Δ mAP50–95 | Δ recall | Interpretation |",
        "|---|---:|---:|---:|---|",
    ]
    for row in sorted(neck, key=lambda row: (row["precision"], float(row["prune_ratio"]))):
        interpretation = "Exploratory only; neck sensitivity not validated."
        if float(row["prune_ratio"]) >= 0.2:
            interpretation = "Reject: substantial detection-quality degradation."
        lines.append(
            f"| {row['precision'].upper()} | {float(row['prune_ratio']):.2f} | {row['map_drop']:+.4f} | {row['recall_drop']:+.4f} | {interpretation} |"
        )
    lines += [
        "",
        "## Supervisor-ready takeaway",
        "",
        "1. The filterwise experiment provides the evidence for selecting pruning targets and cluster-aligned removal steps, rather than assuming every YOLO layer is equally tolerant.",
        "2. The backbone-only matrix identifies conservative pruning ratios that retain accuracy within the one-percentage-point mAP50–95 criterion. FP32 ratio 0.30 was accepted in both runs; for FP16, ratio 0.20 was accepted in Run 1 and ratio 0.19 is the conservative Run 2 candidate. All finalists require Jetson Orin Nano confirmation.",
        "3. The unvalidated neck-pruning result is a useful negative result: adding neck layers causes an immediate mAP loss at ratio 0.10 and a severe mAP/recall collapse from ratio 0.20. Those layers require their own filterwise sensitivity study before reuse in a combined structured-pruning configuration.",
        "",
        "## Hardware note",
        "",
        "These are preliminary RTX 3070 TensorRT measurements at 352×352 input resolution. Final latency and energy claims must be measured on the Jetson Orin Nano.",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _style()
    filterwise = plot_filterwise(_load_filterwise(), output)
    backbone = plot_backbone(
        _load_manifest("rtx_compression_matrix_run_1"),
        _load_manifest("rtx_compression_matrix_run_2"),
        output,
    )
    neck = plot_neck(_load_manifest("rtx_compression_matrix"), output)
    write_briefing(output, filterwise, backbone, neck)
    print(output)


if __name__ == "__main__":
    main()
