"""Export existing pruned checkpoints and compare their ONNX accuracy, size, and latency."""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from infrared_detection.evaluation.artifacts import write_experiment_manifest, write_metrics_csv
from infrared_detection.evaluation.detection_metrics import summarize_detection_metrics


Metrics = dict[str, Any]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _repo_root() / path


def load_comparison_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, Mapping):
        raise ValueError("Pruning-ratio comparison config must be a mapping")
    model = config.get("model")
    baseline = config.get("baseline")
    candidates = config.get("candidates")
    evaluation = config.get("evaluation")
    experiment = config.get("experiment")
    if not isinstance(model, Mapping) or not isinstance(model.get("cluster_size"), int) or model["cluster_size"] <= 0:
        raise ValueError("model.cluster_size must be a positive integer")
    if not isinstance(baseline, Mapping) or not all(isinstance(baseline.get(key), str) for key in ("checkpoint", "onnx")):
        raise ValueError("baseline.checkpoint and baseline.onnx are required")
    if not isinstance(candidates, list):
        raise ValueError("candidates must be a list")
    ratios = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or not isinstance(candidate.get("checkpoint"), str):
            raise ValueError("Each candidate requires a checkpoint path")
        if candidate.get("onnx") is not None and not isinstance(candidate["onnx"], str):
            raise ValueError("Candidate onnx paths must be strings")
        ratio = candidate.get("ratio")
        if not isinstance(ratio, (int, float)) or not 0 < ratio < 1 or ratio in ratios:
            raise ValueError(f"Candidate ratios must be unique numbers between 0 and 1: {ratio!r}")
        ratios.add(ratio)
    if not isinstance(evaluation, Mapping):
        raise ValueError("evaluation must be a mapping")
    if not isinstance(evaluation.get("dataset_yaml"), str):
        raise ValueError("evaluation.dataset_yaml is required")
    if not isinstance(evaluation.get("image_size"), int) or evaluation["image_size"] <= 0:
        raise ValueError("evaluation.image_size must be positive")
    for key in ("confidence", "iou"):
        if not isinstance(evaluation.get(key), (int, float)) or not 0 < evaluation[key] < 1:
            raise ValueError(f"evaluation.{key} must be between 0 and 1")
    if not isinstance(experiment, Mapping) or not isinstance(experiment.get("output_dir"), str):
        raise ValueError("experiment.output_dir is required")
    settings = dict(evaluation)
    settings.setdefault("split", "val")
    settings.setdefault("device", "0")
    settings.setdefault("warmup_runs", 20)
    settings.setdefault("measured_runs", 100)
    settings.setdefault("providers", ["CUDAExecutionProvider", "CPUExecutionProvider"])
    if not isinstance(settings["warmup_runs"], int) or settings["warmup_runs"] < 0:
        raise ValueError("evaluation.warmup_runs must be non-negative")
    if not isinstance(settings["measured_runs"], int) or settings["measured_runs"] <= 0:
        raise ValueError("evaluation.measured_runs must be positive")
    if not isinstance(settings.get("rounds", 1), int) or settings.get("rounds", 1) <= 0:
        raise ValueError("evaluation.rounds must be positive")
    if not isinstance(settings.get("shuffle_seed", 7), int):
        raise ValueError("evaluation.shuffle_seed must be an integer")
    if not isinstance(settings["providers"], list) or not settings["providers"]:
        raise ValueError("evaluation.providers must be a non-empty list")
    config = dict(config)
    config["model"] = dict(model)
    config["baseline"] = dict(baseline)
    config["candidates"] = [dict(candidate) for candidate in candidates]
    config["evaluation"] = settings
    config["experiment"] = dict(experiment)
    config["_config_path"] = str(config_path.resolve())
    return config


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ratio_label(ratio: float) -> str:
    return f"{ratio:g}"


def _candidate_id(cluster_size: int, ratio: float) -> str:
    return f"cluster-{cluster_size}-ratio-{_ratio_label(ratio)}"


@dataclass
class PruningRatioAdapters:
    """Injectable boundaries for model export, validation, and ONNX Runtime timing."""

    export_onnx: Callable[[Path, int, Path], Path]
    evaluate_onnx: Callable[[Path, Mapping[str, Any]], Mapping[str, Any]]
    benchmark_onnx: Callable[[Path, Mapping[str, Any]], Mapping[str, Any]]

    @classmethod
    def defaults(cls) -> "PruningRatioAdapters":
        return cls(_export_checkpoint, _evaluate_onnx, _benchmark_onnx)


def _export_checkpoint(checkpoint: Path, image_size: int, output_dir: Path) -> Path:
    from infrared_detection.evaluation.lpq_export import _export_checkpoint_to_onnx

    return _export_checkpoint_to_onnx(
        checkpoint,
        {"experiment": {"image_size": image_size}},
        output_dir,
    )


def _evaluate_onnx(onnx_path: Path, evaluation: Mapping[str, Any]) -> Mapping[str, Any]:
    from ultralytics import YOLO

    from infrared_detection.evaluation.detection_metrics import evaluate_yolo

    return evaluate_yolo(
        YOLO(str(onnx_path)),
        str(_resolve_path(evaluation["dataset_yaml"])),
        str(evaluation["split"]),
        int(evaluation["image_size"]),
        evaluation["device"],
        float(evaluation["confidence"]),
        float(evaluation["iou"]),
    )


def _benchmark_onnx(onnx_path: Path, evaluation: Mapping[str, Any]) -> Mapping[str, Any]:
    import numpy as np
    import onnxruntime as ort

    image_size = int(evaluation["image_size"])
    sample = np.zeros((1, 3, image_size, image_size), dtype=np.float32)
    session = ort.InferenceSession(str(onnx_path), providers=list(evaluation["providers"]))
    input_name = session.get_inputs()[0].name
    inputs = {input_name: sample}
    for _ in range(int(evaluation["warmup_runs"])):
        session.run(None, inputs)
    timings = []
    for _ in range(int(evaluation["measured_runs"])):
        start = time.perf_counter()
        session.run(None, inputs)
        timings.append((time.perf_counter() - start) * 1000)
    p50, p95 = np.percentile(timings, [50, 95]).tolist()
    return {
        "latency_mean_ms": sum(timings) / len(timings),
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "fps_from_p50": 1000 / p50 if p50 else None,
        "runtime": f"ONNX Runtime {ort.__version__}",
        "providers": session.get_providers(),
        "timed_operation": "session.run only; excludes preprocessing and postprocessing",
        "warmup_runs": int(evaluation["warmup_runs"]),
        "measured_runs": int(evaluation["measured_runs"]),
        "input_shape": [1, 3, image_size, image_size],
    }


def _load_previous_rows(output_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    path = output_dir / "manifest.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            (str(row["variant_id"]), int(row.get("run", 1))): dict(row)
            for row in payload.get("rows", [])
            if isinstance(row, Mapping) and "variant_id" in row
        }
    except (OSError, TypeError, ValueError, AttributeError):
        return {}


def _fingerprint(
    checkpoint: Path | None,
    onnx_path: Path,
    evaluation: Mapping[str, Any],
    row: Mapping[str, Any],
) -> str:
    dataset_path = _resolve_path(evaluation["dataset_yaml"])
    payload = {
        "checkpoint_sha256": _sha256(checkpoint) if checkpoint else None,
        "onnx_sha256": _sha256(onnx_path),
        "dataset_sha256": _sha256(dataset_path),
        "evaluation": dict(evaluation),
    }
    if "run" in row:
        payload.update(run=row["run"], order=row["order"], total=row["total"])
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _matching_export(checkpoint: Path, onnx_path: Path) -> bool:
    source_checkpoint = onnx_path.with_name("export-source.pt")
    checkpoint_hash = _sha256(checkpoint)
    return onnx_path.is_file() and checkpoint_hash is not None and checkpoint_hash == _sha256(source_checkpoint)


def _plan_rows(config: Mapping[str, Any], output_dir: Path) -> list[Metrics]:
    baseline = config["baseline"]
    baseline_checkpoint = _resolve_path(baseline["checkpoint"])
    baseline_onnx = _resolve_path(baseline["onnx"])
    rows = [
        {
            "variant_id": "baseline-fp32",
            "candidate_id": "baseline",
            "prune_ratio": None,
            "onnx_precision": "fp32",
            "checkpoint_path": str(baseline_checkpoint),
            "onnx_path": str(baseline_onnx),
            "status": "planned" if baseline_checkpoint.is_file() else "missing_checkpoint",
            "error": None if baseline_checkpoint.is_file() else f"Checkpoint not found: {baseline_checkpoint}",
        }
    ]
    cluster_size = int(config["model"]["cluster_size"])
    for candidate in config["candidates"]:
        ratio = float(candidate["ratio"])
        candidate_id = _candidate_id(cluster_size, ratio)
        checkpoint = _resolve_path(candidate["checkpoint"])
        output_onnx = output_dir / "onnx" / f"{candidate_id}-fp32" / "model.onnx"
        cached_onnx = _resolve_path(candidate["onnx"]) if candidate.get("onnx") else output_onnx
        exists = checkpoint.is_file()
        rows.append(
            {
                "variant_id": f"{candidate_id}-fp32",
                "candidate_id": candidate_id,
                "prune_ratio": ratio,
                "onnx_precision": "fp32",
                "checkpoint_path": str(checkpoint),
                "onnx_path": str(cached_onnx if _matching_export(checkpoint, cached_onnx) else output_onnx),
                "status": "planned" if exists else "missing_checkpoint",
                "error": None if exists else f"Checkpoint not found: {checkpoint}",
            }
        )
    return rows


def _add_deltas(rows: list[Metrics]) -> None:
    for row in rows:
        if row["candidate_id"] == "baseline" or row.get("status") != "completed":
            continue
        baseline = next(
            (other for other in rows if other["candidate_id"] == "baseline" and other.get("run") == row.get("run")),
            None,
        )
        if not baseline or baseline.get("status") != "completed":
            continue
        for metric in ("map50", "map50_95"):
            row[f"{metric}_delta"] = row.get(metric, 0) - baseline.get(metric, 0)
        baseline_size = baseline.get("onnx_size_bytes", 0)
        row["onnx_size_reduction_percent"] = (
            100 * (baseline_size - row["onnx_size_bytes"]) / baseline_size if baseline_size else None
        )
        baseline_latency = baseline.get("latency_p50_ms", 0)
        row["latency_p50_change_percent"] = (
            100 * (row["latency_p50_ms"] - baseline_latency) / baseline_latency if baseline_latency else None
        )


def _write_results(output_dir: Path, config: Mapping[str, Any], rows: list[Metrics]) -> None:
    _add_deltas(rows)
    write_experiment_manifest(
        output_dir / "manifest.json",
        {
            "config_path": config["_config_path"],
            "model": config["model"],
            "hardware_label": config.get("hardware_label", "rtx3070"),
            "evaluation": config["evaluation"],
            "dataset_sha256": _sha256(_resolve_path(config["evaluation"]["dataset_yaml"])),
            "rows": rows,
        },
    )
    write_metrics_csv(
        output_dir / "results.csv",
        rows,
        leading_columns=("variant_id", "candidate_id", "prune_ratio", "status"),
    )


def run_pruning_ratio_comparison(
    config_path: str | Path,
    *,
    dry_run: bool = False,
    adapters: PruningRatioAdapters | None = None,
) -> list[Metrics]:
    """Compare baseline and configured existing checkpoints; never performs pruning."""

    config = load_comparison_config(config_path)
    output_dir = _resolve_path(config["experiment"]["output_dir"])
    planned = _plan_rows(config, output_dir)
    evaluation = config["evaluation"]
    rows = planned
    if evaluation.get("rounds", 1) > 1:
        rows = []
        rng = random.Random(evaluation.get("shuffle_seed", 7))
        for run in range(1, evaluation["rounds"] + 1):
            remaining = [dict(row) for row in planned]
            rng.shuffle(remaining)
            for order in range(1, len(remaining) + 1):
                row = remaining.pop()
                row.update(run=run, order=order, total=len(planned))
                rows.append(row)
    if dry_run:
        return rows

    output_dir.mkdir(parents=True, exist_ok=True)
    previous = _load_previous_rows(output_dir)
    active = adapters or PruningRatioAdapters.defaults()
    for row in rows:
        if row["status"] == "missing_checkpoint":
            _write_results(output_dir, config, rows)
            continue
        checkpoint = Path(row["checkpoint_path"])
        onnx_path = Path(row["onnx_path"])
        if not _matching_export(checkpoint, onnx_path):
            output_dir_for_export = onnx_path.parent
            try:
                onnx_path = Path(active.export_onnx(checkpoint, int(evaluation["image_size"]), output_dir_for_export))
                row["onnx_path"] = str(onnx_path.resolve())
            except Exception as exc:
                row.update(status="failed", export_error=f"{type(exc).__name__}: {exc}")
                _write_results(output_dir, config, rows)
                continue
        if not onnx_path.is_file():
            row.update(status="missing_artifact", error=f"ONNX artifact not found: {onnx_path}")
            _write_results(output_dir, config, rows)
            continue

        row.update(
            {
                "onnx_size_bytes": onnx_path.stat().st_size,
                "checkpoint_size_bytes": checkpoint.stat().st_size if checkpoint.is_file() else None,
                "checkpoint_sha256": _sha256(checkpoint),
                "onnx_sha256": _sha256(onnx_path),
            }
        )
        fingerprint = _fingerprint(checkpoint if checkpoint.is_file() else None, onnx_path, evaluation, row)
        saved = previous.get((row["variant_id"], row.get("run", 1)), {})
        if saved.get("status") == "completed" and saved.get("fingerprint") == fingerprint:
            row.clear()
            row.update(saved)
            _write_results(output_dir, config, rows)
            continue

        try:
            row.update(summarize_detection_metrics(active.evaluate_onnx(onnx_path, evaluation)))
        except Exception as exc:
            row["evaluation_error"] = f"{type(exc).__name__}: {exc}"
        try:
            row.update(dict(active.benchmark_onnx(onnx_path, evaluation)))
        except Exception as exc:
            row["latency_error"] = f"{type(exc).__name__}: {exc}"
        row["fingerprint"] = fingerprint
        if "evaluation_error" not in row and "latency_error" not in row:
            row["status"] = "completed"
        elif any(key in row for key in ("map50", "latency_p50_ms")):
            row["status"] = "partial"
        else:
            row["status"] = "failed"
        _write_results(output_dir, config, rows)
    return rows
