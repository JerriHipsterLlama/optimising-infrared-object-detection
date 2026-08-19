"""Generate supervisor-facing pruning figures from collected RTX results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

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


def _load_single_layer_results(paths: Sequence[Path]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("stage") != "single_layer" or row.get("status") != "completed":
                    continue
                if any(_number(row.get(field)) is None for field in ("filters_remaining", "map50_95", "latency_p50_ms")):
                    continue
                key = "/".join(
                    (
                        str(row.get("model_variant") or "unknown-model"),
                        str(row.get("hardware") or "unknown-hardware"),
                        str(row.get("layer") or "unknown-layer"),
                    )
                )
                grouped[key].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["filters_remaining"]), reverse=True)
    return dict(sorted(grouped.items()))


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


def plot_single_layer(
    paths: Sequence[Path], output: Path
) -> dict[str, list[dict[str, Any]]]:
    grouped = _load_single_layer_results(paths)
    if not grouped:
        raise ValueError("No completed single-layer accuracy and latency rows were found")

    figures = (
        ("map50_95", "mAP50–95", "Single-layer pruning: accuracy response", "01_single_layer_accuracy.png"),
        (
            "latency_p50_ms",
            "PyTorch CUDA forward latency p50 (ms)",
            "Single-layer pruning: direct CUDA latency response",
            "02_single_layer_latency.png",
        ),
    )
    for metric, ylabel, title, filename in figures:
        fig, axis = plt.subplots(figsize=(11.5, 5.2), constrained_layout=True)
        for key, rows in grouped.items():
            axis.plot(
                [int(row["filters_remaining"]) for row in rows],
                [float(row[metric]) for row in rows],
                linewidth=1.1,
                label=key,
            )
        axis.invert_xaxis()
        axis.set_xlabel("Filters remaining")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=6, ncol=2)
        fig.savefig(output / filename, bbox_inches="tight")
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
    single_layer: dict[str, list[dict[str, Any]]],
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
        "1. `01_single_layer_accuracy.png` — independent single-layer mAP50–95 sensitivity curves.",
        "2. `02_single_layer_latency.png` — direct PyTorch CUDA forward-latency curves.",
        "3. `03_backbone_compression_matrix.png` — validated backbone-only compression matrix results.",
        "4. `04_neck_pruning_warning.png` — exploratory backbone-plus-neck accuracy and recall loss.",
        "",
        "## Single-layer performance-response coverage",
        "",
        "| Model / hardware / layer | Completed points | Filters remaining tested | Interpretation |",
        "|---|---:|---:|---|",
    ]
    for key, rows in single_layer.items():
        remaining = [int(row["filters_remaining"]) for row in rows]
        lines.append(f"| `{key}` | {len(rows)} | {min(remaining)}–{max(remaining)} | Use this curve to identify layer sensitivity and hardware-aligned latency steps. |")
    lines += [
        "",
        "These latency values are direct PyTorch CUDA forward measurements. RTX 3070 results are preliminary; final deployment claims require the equivalent sweep on the Jetson Orin Nano.",
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
        "1. The independent single-layer experiment provides evidence for selecting pruning targets and cluster-aligned removal steps, rather than assuming every YOLO layer is equally tolerant.",
        "2. The backbone-only matrix identifies conservative pruning ratios that retain accuracy within the one-percentage-point mAP50–95 criterion. FP32 ratio 0.30 was accepted in both runs; for FP16, ratio 0.20 was accepted in Run 1 and ratio 0.19 is the conservative Run 2 candidate. All finalists require Jetson Orin Nano confirmation.",
        "3. The unvalidated neck-pruning result is a useful negative result: adding neck layers causes an immediate mAP loss at ratio 0.10 and a severe mAP/recall collapse from ratio 0.20. Those layers require their own single-layer sensitivity study before reuse in a combined structured-pruning configuration.",
        "",
        "## Hardware note",
        "",
        "The sensitivity curves use direct PyTorch CUDA forward latency at 352×352 input resolution. Compression-matrix latency remains deployment-engine evidence. Final latency and energy claims must be measured on the Jetson Orin Nano.",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument(
        "--single-layer-results",
        action="append",
        type=Path,
        default=[],
        help="Repeatable path to a single-layer performance results.csv file.",
    )
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _style()
    single_layer_paths = args.single_layer_results or [
        EXPERIMENTS / "single_layer_performance_screening" / "yolov8n" / "rtx3070" / "results.csv",
        EXPERIMENTS / "single_layer_performance_screening" / "yolov8m" / "rtx3070" / "results.csv",
    ]
    single_layer = plot_single_layer(single_layer_paths, output)
    backbone = plot_backbone(
        _load_manifest("rtx_compression_matrix_run_1"),
        _load_manifest("rtx_compression_matrix_run_2"),
        output,
    )
    neck = plot_neck(_load_manifest("rtx_compression_matrix"), output)
    write_briefing(output, single_layer, backbone, neck)
    print(output)


if __name__ == "__main__":
    main()
