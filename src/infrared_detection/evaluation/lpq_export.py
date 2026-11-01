"""Export a selected PyTorch checkpoint to precision-specific ONNX artifacts."""

from __future__ import annotations

import json
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from infrared_detection.evaluation.artifacts import write_experiment_manifest, write_metrics_csv


OFFICIAL_PRECISIONS = ("fp32", "fp16", "int8")
Metrics = dict[str, Any]


def _validate_precisions(precisions: Any) -> list[str]:
    if precisions not in (list(OFFICIAL_PRECISIONS[:2]), list(OFFICIAL_PRECISIONS)):
        raise ValueError(
            "Precision matrix must be the ordered list [fp32, fp16] or "
            "[fp32, fp16, int8]"
        )
    return list(precisions)


def load_lpq_config(path: str | Path) -> dict[str, Any]:
    """Load an LPQ-only YAML config and validate its precision variants."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("LPQ config must be a mapping.")
    if "pruning" in config:
        raise ValueError("Pruning is a separate cluster-pruning workflow; remove the pruning section from this config")
    _validate_precisions(config.get("precisions"))
    config["_config_path"] = str(config_path.resolve())
    return config


def planned_lpq_variants(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return LPQ precision variants for the one explicitly selected checkpoint."""
    precisions = _validate_precisions(config.get("precisions"))
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("LPQ config requires model.variant_id")
    candidate_id = model.get("variant_id")
    if not isinstance(candidate_id, str) or not candidate_id or Path(candidate_id).name != candidate_id:
        raise ValueError("model.variant_id must be a non-empty directory-safe name")
    return [
        {
            "variant_id": f"{candidate_id}-{precision}",
            "candidate_id": candidate_id,
            "compression": "lpq",
            "precision": precision,
            "status": "planned",
            "error": None,
            "provenance": {"planner": "lpq_export", "stage": "planning"},
        }
        for precision in precisions
    ]


def write_lpq_manifest(
    output_dir: Path,
    rows: list[Mapping[str, Any]],
    *,
    metadata: Mapping[str, Any] | None = None,
    preserve_existing: bool = True,
) -> None:
    """Write manifest and CSV checkpoints while retaining prior partial rows."""

    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    existing_rows: dict[str, dict[str, Any]] = {}
    existing_metadata: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping):
                existing_metadata = {key: value for key, value in payload.items() if key != "rows"}
            existing_rows = {
                str(row["variant_id"]): dict(row)
                for row in payload.get("rows", [])
                if isinstance(row, Mapping) and "variant_id" in row
            }
        except (OSError, TypeError, ValueError, AttributeError):
            existing_rows = {}

    if not preserve_existing:
        existing_rows = {}
    for row in rows:
        if "variant_id" not in row:
            raise ValueError("Every LPQ manifest row requires variant_id")
        existing_rows[str(row["variant_id"])] = dict(row)

    merged_rows = list(existing_rows.values())
    payload = {**existing_metadata, **dict(metadata or {}), "rows": merged_rows}
    write_experiment_manifest(manifest_path, payload)
    write_metrics_csv(output_dir / "results.csv", merged_rows)


@dataclass
class LPQExportAdapters:
    """Injectable side-effect boundary for ONNX artifact generation."""

    export_checkpoint: Callable[[Path, Mapping[str, Any], Path], Path]
    convert_precision: Callable[[Path, str, Path, Mapping[str, Any]], Mapping[str, Any]]

    @classmethod
    def defaults(cls, config: Mapping[str, Any]) -> "LPQExportAdapters":
        """Create real adapters lazily, so dry runs require no ML runtime."""

        return cls(
            export_checkpoint=_export_checkpoint_to_onnx,
            convert_precision=_convert_precision_onnx,
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _repo_root() / path


def _lpq_output_dir(config: Mapping[str, Any]) -> Path:
    experiment = config.get("experiment")
    if not isinstance(experiment, Mapping) or not isinstance(experiment.get("output_dir"), str):
        raise ValueError("LPQ export requires experiment.output_dir")
    return _resolve_repo_path(experiment["output_dir"])


def _selected_checkpoint(config: Mapping[str, Any]) -> Path:
    model = config.get("model")
    if not isinstance(model, Mapping) or not isinstance(model.get("checkpoint"), str):
        raise ValueError("LPQ export requires model.checkpoint")
    return _resolve_repo_path(model["checkpoint"])


def _row_provenance(config_path: Path, checkpoint: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "config_path": str(config_path.resolve()),
        "source_checkpoint": str(checkpoint.resolve()),
        "config_fingerprint": hashlib.sha256(
            json.dumps(
                {key: value for key, value in config.items() if key != "_config_path"},
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest(),
    }


def _load_completed_rows(output_dir: Path, rows: list[Metrics]) -> list[Metrics]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return rows
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        saved_rows = {
            str(row["variant_id"]): row
            for row in payload.get("rows", [])
            if isinstance(row, Mapping) and isinstance(row.get("variant_id"), str)
        }
    except (OSError, TypeError, ValueError, AttributeError):
        return rows
    for row in rows:
        saved = saved_rows.get(str(row["variant_id"]))
        if saved is not None and _is_resumable_completed_row(saved, row):
            row.clear()
            row.update(dict(saved))
    return rows


def _is_resumable_completed_row(saved: Mapping[str, Any], planned: Mapping[str, Any]) -> bool:
    if saved.get("status") != "completed" or saved.get("provenance") != planned.get("provenance"):
        return False
    paths = (saved.get("base_onnx_path"), saved.get("onnx_path"))
    if any(not isinstance(path, str) or not Path(path).is_file() for path in paths):
        return False
    checkpoint = saved.get("checkpoint_path")
    return isinstance(checkpoint, str) and Path(checkpoint).is_file()


def _failure(row: Metrics, exc: Exception) -> None:
    row["status"] = "failed"
    row["error"] = f"{type(exc).__name__}: {exc}"


def _write_lpq_artifacts(
    output_dir: Path, config_path: Path, config: Mapping[str, Any], rows: list[Metrics]
) -> None:
    write_lpq_manifest(
        output_dir,
        rows,
        metadata={
            "config_path": str(config_path.resolve()),
            "candidate_count": len({str(row["candidate_id"]) for row in rows}),
            "precision_count": len(rows),
        },
        preserve_existing=False,
    )


def run_lpq_export(
    config_path: str | Path,
    dry_run: bool = False,
    adapters: LPQExportAdapters | None = None,
) -> list[Metrics]:
    """Export one selected checkpoint to FP32/FP16/INT8 ONNX variants."""

    config_file = Path(config_path).resolve()
    config = load_lpq_config(config_file)
    output_dir = _lpq_output_dir(config)
    rows = planned_lpq_variants(config)
    checkpoint = _selected_checkpoint(config)
    common_provenance = _row_provenance(config_file, checkpoint, config)
    for row in rows:
        row["provenance"] = {
            **dict(row["provenance"]),
            **common_provenance,
            "candidate_id": row["candidate_id"],
            "precision": row["precision"],
        }
    if dry_run:
        _write_lpq_artifacts(output_dir, config_file, config, rows)
        return rows

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected model.checkpoint does not exist: {checkpoint}")
    active = adapters or LPQExportAdapters.defaults(config)
    rows = _load_completed_rows(output_dir, rows)
    candidate_id = str(rows[0]["candidate_id"])
    candidate_dir = output_dir / candidate_id
    completed_rows = [row for row in rows if row.get("status") == "completed"]
    reusable = next(
        (row for row in completed_rows if Path(str(row.get("base_onnx_path", ""))).is_file()),
        None,
    )
    try:
        if reusable is not None:
            source_onnx = Path(str(reusable["base_onnx_path"]))
        else:
            source_onnx = Path(active.export_checkpoint(checkpoint, config, candidate_dir / "source"))
            if not source_onnx.is_file():
                raise FileNotFoundError(f"Base ONNX export was not written: {source_onnx}")
    except Exception as exc:
        for row in rows:
            if row.get("status") != "completed":
                _failure(row, exc)
        _write_lpq_artifacts(output_dir, config_file, config, rows)
        return rows

    for row in rows:
        if row.get("status") == "completed":
            continue
        precision = str(row["precision"])
        onnx = candidate_dir / precision / "model.onnx"
        try:
            conversion = dict(active.convert_precision(source_onnx, precision, onnx, config))
            if not onnx.is_file():
                raise FileNotFoundError(f"Precision ONNX was not written: {onnx}")
            row.update(
                {
                    "checkpoint_path": str(checkpoint.resolve()),
                    "base_onnx_path": str(source_onnx.resolve()),
                    "base_onnx_size_bytes": source_onnx.stat().st_size,
                    "onnx_path": str(onnx.resolve()),
                    "onnx_size_bytes": onnx.stat().st_size,
                    "conversion": conversion,
                    "status": "completed",
                    "error": None,
                }
            )
        except Exception as exc:
            _failure(row, exc)
        finally:
            _write_lpq_artifacts(output_dir, config_file, config, rows)
    return rows


def _export_checkpoint_to_onnx(checkpoint: Path, config: Mapping[str, Any], output_dir: Path) -> Path:
    from infrared_detection.evaluation.single_layer_performance_screening import (
        _install_legacy_pathlib_checkpoint_compatibility,
    )

    _install_legacy_pathlib_checkpoint_compatibility()
    from infrared_detection.export import export_yolo

    output_dir.mkdir(parents=True, exist_ok=True)
    staged_checkpoint = output_dir / "export-source.pt"
    shutil.copy2(checkpoint, staged_checkpoint)
    exported = Path(export_yolo(staged_checkpoint, format="onnx", imgsz=int(config.get("experiment", {}).get("image_size", 640))))
    if not exported.is_file():
        raise FileNotFoundError(f"ONNX export was not written: {exported}")
    isolated = output_dir / "model.onnx"
    if exported.resolve() != isolated.resolve():
        shutil.copy2(exported, isolated)
    return isolated


def _convert_precision_onnx(
    source: Path, precision: str, output: Path, config: Mapping[str, Any]
) -> Mapping[str, Any]:
    from infrared_detection.compression.quantization.onnx_precision import (
        calibration_reader_from_onnx,
        prepare_precision_onnx,
    )

    quantization = config.get("quantization", {})
    data = config.get("data", {})
    if not isinstance(quantization, Mapping) or not isinstance(data, Mapping):
        raise ValueError("quantization and data config sections must be mappings")
    reader = None
    calibration_method = str(quantization.get("calibration_method", "entropy")).lower()
    if precision == "int8":
        calibration_dir = data.get("calibration_image_dir")
        if not isinstance(calibration_dir, str) or not calibration_dir:
            raise ValueError("INT8 conversion requires data.calibration_image_dir")
        calibration_split = data.get("calibration_split", data.get("split"))
        reader = calibration_reader_from_onnx(
            source,
            _resolve_repo_path(calibration_dir),
            sample_limit=int(quantization.get("calibration_samples", 128)),
            batch_size=int(quantization.get("batch_size", 1)),
            calibration_split=str(calibration_split) if calibration_split is not None else None,
        )
    return prepare_precision_onnx(
        source,
        output,
        precision,
        calibration_reader=reader,
        calibration_method=calibration_method,
    )
