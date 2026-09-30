"""Directory-driven quality and target-runtime testing for precision ONNX artifacts."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from infrared_detection.evaluation.artifacts import write_experiment_manifest, write_metrics_csv
from infrared_detection.evaluation.detection_metrics import summarize_detection_metrics


_PRECISIONS = {"fp32", "fp16", "int8"}
_BACKENDS = {"python", "jetson_cpp", "both"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _repo_root() / path


def load_testing_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the directory-driven model testing configuration."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Model testing config must be a mapping")
    models = config.get("models")
    testing = config.get("testing")
    if not isinstance(models, Mapping) or not isinstance(models.get("directory"), str):
        raise ValueError("Model testing config requires models.directory")
    if not isinstance(models.get("pattern", "**/model.onnx"), str):
        raise ValueError("models.pattern must be a string")
    if not isinstance(testing, Mapping):
        raise ValueError("Model testing config requires a testing mapping")
    backend = testing.get("backend")
    if backend not in _BACKENDS:
        raise ValueError(f"testing.backend must be one of {sorted(_BACKENDS)}")
    if not isinstance(testing.get("dataset_yaml"), str) or not testing["dataset_yaml"]:
        raise ValueError("testing.dataset_yaml is required")
    if not isinstance(testing.get("split"), str) or not testing["split"]:
        raise ValueError("testing.split is required")
    if not isinstance(testing.get("image_size"), int) or testing["image_size"] <= 0:
        raise ValueError("testing.image_size must be positive")
    for key in ("confidence", "iou"):
        value = testing.get(key)
        if not isinstance(value, (int, float)) or not 0 < value < 1:
            raise ValueError(f"testing.{key} must be between 0 and 1")
    if not isinstance(testing.get("output_dir"), str) or not testing["output_dir"]:
        raise ValueError("testing.output_dir is required")
    if not isinstance(testing.get("workspace_mb", 4096), int) or testing.get("workspace_mb", 4096) <= 0:
        raise ValueError("testing.workspace_mb must be positive")
    if backend in {"jetson_cpp", "both"}:
        if not isinstance(testing.get("runner"), str) or not testing["runner"]:
            raise ValueError("Jetson backend requires testing.runner")
        if not isinstance(testing.get("image_dir"), str) or not testing["image_dir"]:
            raise ValueError("Jetson backend requires testing.image_dir")
        if not isinstance(testing.get("iterations", 100), int) or testing.get("iterations", 100) <= 0:
            raise ValueError("testing.iterations must be positive")
        if not isinstance(testing.get("warmup", 20), int) or testing.get("warmup", 20) < 0:
            raise ValueError("testing.warmup must be non-negative")
    config["models"] = {**models, "pattern": models.get("pattern", "**/model.onnx")}
    config["testing"] = dict(testing)
    config["_config_path"] = str(config_path.resolve())
    return config


def discover_onnx_models(directory: Path, pattern: str) -> list[Path]:
    """Return sorted ONNX models matching the configured glob, excluding source exports."""

    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"ONNX artifact directory does not exist: {root}")
    return sorted(
        path.resolve()
        for path in root.glob(pattern)
        if path.is_file() and path.suffix.lower() == ".onnx" and path.parent.name.lower() != "source"
    )


def _artifact_identity(path: Path) -> tuple[str, str]:
    artifact_label = path.parent.name.lower()
    if artifact_label in _PRECISIONS:
        precision = artifact_label
        candidate_id = path.parent.parent.name
    else:
        candidate_id, separator, precision = artifact_label.rpartition("-")
        if not separator or precision not in _PRECISIONS:
            precision = ""
        else:
            # Older compression runs stored one artifact per '<candidate>-<precision>' directory.
            candidate_id = path.parent.name[: -(len(precision) + 1)]
    if precision not in _PRECISIONS:
        raise ValueError(
            f"Cannot determine a supported precision for ONNX artifact {path}; "
            "expected an fp32/fp16/int8 parent directory or an exact '<candidate>-<precision>' directory"
        )
    if not candidate_id or candidate_id in {".", ".."}:
        raise ValueError(f"Cannot determine candidate identity for ONNX artifact {path}")
    return candidate_id, precision


def _onnx_inventory(path: Path) -> dict[str, Any]:
    """Capture graph opsets and external-weight files without loading weight payloads."""

    import onnx

    model = onnx.load(str(path), load_external_data=False)
    locations = sorted(
        {
            entry.value
            for initializer in model.graph.initializer
            for entry in initializer.external_data
            if entry.key == "location"
        }
    )
    sidecars = [(path.parent / location).resolve() for location in locations]
    missing = [sidecar for sidecar in sidecars if not sidecar.is_file()]
    if missing:
        raise FileNotFoundError(
            f"ONNX artifact {path} references missing external-weight sidecar(s): "
            + ", ".join(str(sidecar) for sidecar in missing)
        )
    return {
        "onnx_opsets": {entry.domain or "ai.onnx": int(entry.version) for entry in model.opset_import},
        "external_data_files": [str(sidecar) for sidecar in sidecars],
    }


def planned_test_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Discover artifacts and validate their explicit candidate/precision identity."""

    models = config["models"]
    paths = discover_onnx_models(_resolve_path(models["directory"]), models.get("pattern", "**/model.onnx"))
    if not paths:
        raise ValueError(
            f"No ONNX artifacts matched {models.get('pattern', '**/model.onnx')!r} "
            f"under {_resolve_path(models['directory'])}"
        )
    rows = []
    for onnx_path in paths:
        candidate_id, precision = _artifact_identity(onnx_path)
        row = {
                "variant_id": f"{candidate_id}-{precision}",
                "candidate_id": candidate_id,
                "precision": precision,
                "onnx_path": str(onnx_path),
                "status": "planned",
                "python_status": "not_requested",
                "jetson_cpp_status": "not_requested",
                "error": None,
            }
        try:
            row.update(_onnx_inventory(onnx_path))
        except Exception as exc:
            row["artifact_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


@dataclass
class ModelTestingMatrixAdapters:
    """Injectable execution boundary for quality, engine, and Jetson tests."""

    evaluate_python: Callable[[Path, Mapping[str, Any]], Mapping[str, Any]]
    build_engine: Callable[[Path, Path, Mapping[str, Any]], Mapping[str, Any]]
    benchmark_jetson: Callable[[Path, Mapping[str, Any]], Mapping[str, Any]]

    @classmethod
    def defaults(cls) -> "ModelTestingMatrixAdapters":
        return cls(_evaluate_python, _build_local_engine, _benchmark_jetson_cpp)


def _evaluate_python(onnx_path: Path, config: Mapping[str, Any]) -> Mapping[str, Any]:
    from ultralytics import YOLO

    from infrared_detection.evaluation.detection_metrics import evaluate_yolo

    testing = config["testing"]
    model = YOLO(str(onnx_path))
    return evaluate_yolo(
        model,
        str(_resolve_path(testing["dataset_yaml"])),
        str(testing["split"]),
        int(testing["image_size"]),
        testing.get("device", "0"),
        float(testing["confidence"]),
        float(testing["iou"]),
    )


def _build_local_engine(
    onnx_path: Path, engine_path: Path, config: Mapping[str, Any]
) -> Mapping[str, Any]:
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    trtexec = shutil.which("trtexec")
    if trtexec is None:
        raise FileNotFoundError("Target-local engine build requires 'trtexec' on PATH")
    workspace_mb = int(config["testing"].get("workspace_mb", 4096))
    command = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{workspace_mb}",
        "--stronglyTyped",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    if not engine_path.is_file():
        raise FileNotFoundError(f"TensorRT completed without writing engine for {onnx_path}: {engine_path}")
    return {
        "command": command,
        "precision": engine_path.parent.name,
        "onnx_path": str(onnx_path.resolve()),
        "engine_path": str(engine_path.resolve()),
        "engine_size_bytes": engine_path.stat().st_size,
        "workspace_mb": workspace_mb,
        "strongly_typed": True,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _benchmark_jetson_cpp(engine_path: Path, config: Mapping[str, Any]) -> Mapping[str, Any]:
    testing = config["testing"]
    runner = _resolve_path(testing["runner"])
    if not runner.is_file():
        raise FileNotFoundError(f"Jetson C++ benchmark runner does not exist: {runner}")
    output_path = engine_path.with_name("benchmark.json")
    command = [
        str(runner),
        "--engine", str(engine_path),
        "--input-dir", str(_resolve_path(testing["image_dir"])),
        "--output-json", str(output_path),
        "--warmup", str(int(testing.get("warmup", 20))),
        "--iterations", str(int(testing.get("iterations", 100))),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    if not output_path.is_file():
        raise FileNotFoundError(f"Jetson C++ benchmark did not write results: {output_path}")
    metrics = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, Mapping):
        raise ValueError(f"Jetson C++ benchmark output is not a JSON object: {output_path}")
    return {**dict(metrics), "runner": str(runner), "command": command, "stdout": completed.stdout, "stderr": completed.stderr}


def _write_results(output_dir: Path, config: Mapping[str, Any], rows: list[dict[str, Any]]) -> None:
    write_experiment_manifest(
        output_dir / "manifest.json",
        {
            "config_path": config.get("_config_path"),
            "artifact_directory": str(_resolve_path(config["models"]["directory"])),
            "backend": config["testing"]["backend"],
            "hardware_label": config["testing"].get("hardware_label"),
            "rows": rows,
        },
    )
    write_metrics_csv(output_dir / "results.csv", rows, leading_columns=("variant_id", "candidate_id", "precision", "status"))


def _row_status(row: Mapping[str, Any], backend: str) -> str:
    statuses = []
    if backend in {"python", "both"}:
        statuses.append(row["python_status"])
    if backend in {"jetson_cpp", "both"}:
        statuses.append(row["jetson_cpp_status"])
    if all(status == "completed" for status in statuses):
        return "completed"
    if any(status == "completed" for status in statuses):
        return "partial"
    return "failed"


def run_model_testing_matrix(
    config_path: str | Path,
    dry_run: bool = False,
    adapters: ModelTestingMatrixAdapters | None = None,
) -> list[dict[str, Any]]:
    """Evaluate all discovered ONNX artifacts and preserve independent backend results."""

    config = load_testing_config(config_path)
    rows = planned_test_rows(config)
    output_dir = _resolve_path(config["testing"]["output_dir"])
    if dry_run:
        return rows

    active = adapters or ModelTestingMatrixAdapters.defaults()
    backend = str(config["testing"]["backend"])
    hardware = str(config["testing"].get("hardware_label", "target"))
    for row in rows:
        onnx_path = Path(row["onnx_path"])
        if row.get("artifact_error"):
            row["status"] = "failed"
            row["error"] = row["artifact_error"]
            _write_results(output_dir, config, rows)
            continue
        if backend in {"python", "both"}:
            try:
                row["python_metrics"] = dict(
                    summarize_detection_metrics(active.evaluate_python(onnx_path, config))
                )
                row["python_status"] = "completed"
            except Exception as exc:
                row["python_status"] = "failed"
                row["python_error"] = f"{type(exc).__name__}: {exc}"
        if backend in {"jetson_cpp", "both"}:
            engine_path = (
                output_dir / hardware / row["candidate_id"] / row["precision"] / "model.engine"
            )
            try:
                build_info = dict(active.build_engine(onnx_path, engine_path, config))
                engine_path = Path(str(build_info.get("engine_path", engine_path)))
                jetson_metrics = dict(active.benchmark_jetson(engine_path, config))
                row.update(
                    {
                        "engine_path": str(engine_path.resolve()),
                        "engine_size_bytes": build_info.get("engine_size_bytes"),
                        "engine_build": build_info,
                        "jetson_cpp_metrics": jetson_metrics,
                        "jetson_cpp_status": "completed",
                    }
                )
            except Exception as exc:
                row["jetson_cpp_status"] = "failed"
                row["jetson_cpp_error"] = f"{type(exc).__name__}: {exc}"
        row["status"] = _row_status(row, backend)
        if row["status"] == "failed":
            row["error"] = row.get("python_error") or row.get("jetson_cpp_error")
        _write_results(output_dir, config, rows)
    return rows
