"""Deterministic planning primitives for single-layer performance screening."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
import hashlib
import importlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch import nn
import yaml

from infrared_detection.compression.pruning import rank_filters_by_minimum_weight
from infrared_detection.evaluation.artifacts import write_metrics_csv


_PRODUCTION_DEPENDENCY_MODULES = {
    "run_filterwise_probe": "infrared_detection.compression.pruning.cluster_probe",
    "benchmark_pytorch_cuda_forward": "infrared_detection.benchmarking.pytorch_cuda",
    "collect_model_stats": "infrared_detection.evaluation.model_stats",
}


def _production_dependency(name: str):
    existing = globals().get(name)
    if existing is not None:
        return existing
    module_name = _PRODUCTION_DEPENDENCY_MODULES.get(name)
    if module_name is None:
        raise RuntimeError(f"Required production helper {name!r} is unavailable")
    candidate = getattr(importlib.import_module(module_name), name)
    globals()[name] = candidate
    return candidate


@dataclass(frozen=True)
class FilterRanking:
    original_index: int
    score: float


@dataclass(frozen=True)
class LayerPlan:
    name: str
    filters_before: int
    ranking: tuple[FilterRanking, ...]


@dataclass(frozen=True)
class ResumeState:
    version: int
    fingerprint: str
    active_layer: str | None
    next_filter_rank: int
    remaining_original_indices: tuple[int, ...]
    last_candidate_id: str | None


@dataclass(frozen=True)
class ScreeningAdapters:
    @classmethod
    def defaults(cls, config: dict) -> "ScreeningAdapters":
        """Build the real YOLO/CUDA adapters without importing optional services at module import time."""
        import torch

        configured_device = str(config.get("runtime", {}).get("device", "0"))
        if configured_device in {"0", "cuda", "cuda:0"} and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; single-layer performance screening requires CUDA")

        def cuda_device(settings: dict) -> str:
            device = str(settings["runtime"].get("device", "0"))
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; single-layer performance screening requires CUDA")
            return device if device.startswith("cuda:") else f"cuda:{device}"

        def unwrap(wrapper):
            return getattr(wrapper, "model", wrapper)

        def load_model(checkpoint):
            from ultralytics import YOLO

            cuda_device(config)
            return YOLO(str(checkpoint))

        def evaluate(wrapper, settings: dict) -> dict:
            cuda_device(settings)
            runtime = settings["runtime"]
            experiment = settings["experiment"]
            data = settings["data"]
            result = wrapper.val(
                data=data["dataset_yaml"], split=data.get("split", "val"), imgsz=experiment["image_size"],
                device=runtime["device"], conf=runtime["conf"], iou=runtime["iou"],
                batch=experiment.get("batch_size", 1), half=runtime.get("precision") == "fp16", verbose=False,
            )
            box = result.box
            return {
                "map50_95": float(box.map), "map50": float(box.map50),
                "precision": float(box.mp), "recall": float(box.mr), "per_class_ap": list(box.maps),
            }

        def prune_filter(wrapper, layer: str, physical_filter_index: int, settings: dict):
            example_input = torch.randn(1, 3, settings["experiment"]["image_size"], settings["experiment"]["image_size"])
            wrapper.model = _production_dependency("run_filterwise_probe")(
                unwrap(wrapper), example_input, layer, (physical_filter_index,)
            )
            return wrapper

        def profile(wrapper, settings: dict) -> dict:
            device = cuda_device(settings)
            precision = settings["runtime"].get("precision", "fp32")
            model = unwrap(wrapper).to(device)
            if precision == "fp16":
                model = model.half()
            example_input = torch.randn(1, 3, settings["experiment"]["image_size"], settings["experiment"]["image_size"])
            example_input = example_input.to(device)
            if precision == "fp16":
                example_input = example_input.half()
            return _production_dependency("benchmark_pytorch_cuda_forward")(
                model, example_input, warmup=settings["screening"]["warmup"], iterations=settings["screening"]["iterations"],
            )

        return cls(
            load_model=load_model,
            evaluate=evaluate,
            prune_filter=prune_filter,
            profile=profile,
            stats=lambda wrapper: _production_dependency("collect_model_stats")(unwrap(wrapper)),
            save_checkpoint=lambda wrapper, path: wrapper.save(path),
        )
    """Injectable model operations used by the screening workflow."""

    load_model: Callable[[str | Path], Any]
    evaluate: Callable[[Any, Mapping[str, Any]], Mapping[str, Any]]
    prune_filter: Callable[[Any, str, int, Mapping[str, Any]], Any]
    profile: Callable[[Any, Mapping[str, Any]], Mapping[str, Any]]
    stats: Callable[[Any], Mapping[str, Any]]
    save_checkpoint: Callable[[Any, str | Path], None]


_RESEARCH_FIELDS = (
    "candidate_id", "model_variant", "hardware", "status", "layer", "filter_rank",
    "original_filter_index", "physical_filter_index", "minimum_weight_score", "filters_before",
    "filters_removed", "filters_remaining", "map50_95", "map50_95_drop", "map50", "precision",
    "recall", "latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "fps",
)
_CSV_TRAILING_FIELDS = ("reason", "error")


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace a small text artifact using a sibling temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def experiment_fingerprint(
    config: Mapping[str, Any], checkpoint_path: str | Path, dataset_path: str | Path
) -> str:
    """Return a stable identifier for the inputs that define a screening run."""

    checkpoint = Path(checkpoint_path)
    dataset = Path(dataset_path)
    experiment = config["experiment"]
    runtime = config["runtime"]
    screening = config["screening"]
    payload = {
        "checkpoint": {"path": str(checkpoint.resolve()), "size": checkpoint.stat().st_size, "mtime_ns": checkpoint.stat().st_mtime_ns},
        "dataset": {"path": str(dataset.resolve()), "size": dataset.stat().st_size, "mtime_ns": dataset.stat().st_mtime_ns},
        "model_variant": experiment["model_variant"],
        "hardware": experiment["hardware_label"],
        "image_size": experiment["image_size"],
        "precision": runtime["precision"],
        "layer_patterns": screening["layer_patterns"],
        "ranking_version": "minimum-weight-mean-square-v1",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_screening_artifacts(
    output_dir: str | Path,
    config_path: str | Path,
    rows: list[Mapping[str, Any]],
    rankings: Mapping[str, Any],
    state: ResumeState | None,
) -> None:
    """Persist screening progress so it can be resumed without ambiguity."""

    directory = Path(output_dir)
    normalized_rows = [
        {**{field: None for field in (*_RESEARCH_FIELDS, *_CSV_TRAILING_FIELDS)}, **dict(row)}
        for row in rows
    ]
    write_metrics_csv(
        directory / "results.csv",
        normalized_rows,
        leading_columns=_RESEARCH_FIELDS,
        trailing_columns=_CSV_TRAILING_FIELDS,
    )
    _atomic_write_json(
        directory / "manifest.json",
        {"config_path": str(Path(config_path).resolve()), "fingerprint": state.fingerprint if state else None, "rows": normalized_rows},
    )
    _atomic_write_json(directory / "rankings.json", dict(rankings))
    _atomic_write_json(
        directory / "state.json",
        None if state is None else {
            "version": state.version,
            "fingerprint": state.fingerprint,
            "active_layer": state.active_layer,
            "next_filter_rank": state.next_filter_rank,
            "remaining_original_indices": list(state.remaining_original_indices),
            "last_candidate_id": state.last_candidate_id,
        },
    )


def load_screening_artifacts(
    output_dir: str | Path, expected_fingerprint: str
) -> tuple[list[dict], dict, ResumeState | None]:
    """Load persisted screening artifacts, rejecting a different experiment's state."""

    directory = Path(output_dir)
    manifest_path = directory / "manifest.json"
    rankings_path = directory / "rankings.json"
    state_path = directory / "state.json"
    if not manifest_path.exists() and not rankings_path.exists() and not state_path.exists():
        return [], {}, None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    rankings = json.loads(rankings_path.read_text(encoding="utf-8")) if rankings_path.exists() else {}
    serialized_state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
    rows = [dict(row) for row in manifest.get("rows", [])]
    if serialized_state is None:
        return rows, dict(rankings), None
    if serialized_state.get("fingerprint") != expected_fingerprint:
        raise ValueError("experiment fingerprint does not match saved screening artifacts")
    state = ResumeState(
        version=int(serialized_state["version"]),
        fingerprint=str(serialized_state["fingerprint"]),
        active_layer=serialized_state.get("active_layer"),
        next_filter_rank=int(serialized_state["next_filter_rank"]),
        remaining_original_indices=tuple(int(index) for index in serialized_state["remaining_original_indices"]),
        last_candidate_id=serialized_state.get("last_candidate_id"),
    )
    return rows, dict(rankings), state


def save_pending_checkpoint(
    model: Any, output_dir: str | Path, save_checkpoint: Callable[[Any, str | Path], None] | None = None
) -> Path:
    """Save a candidate checkpoint without disturbing the last valid resume point."""

    pending_path = Path(output_dir) / "resume.pending.pt"
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    save = save_checkpoint if save_checkpoint is not None else getattr(model, "save", None)
    if not callable(save):
        raise RuntimeError("Resume checkpoint requires a save-capable model or checkpoint adapter.")
    if save_checkpoint is None:
        save(pending_path)
    else:
        save(model, pending_path)
    if not pending_path.is_file() or pending_path.stat().st_size == 0:
        raise RuntimeError("Pending resume checkpoint was not written or is empty.")
    return pending_path


def promote_pending_checkpoint(output_dir: str | Path) -> Path:
    """Atomically promote the successfully measured pending resume checkpoint."""

    directory = Path(output_dir)
    pending_path = directory / "resume.pending.pt"
    resume_path = directory / "resume.pt"
    if not pending_path.is_file() or pending_path.stat().st_size == 0:
        raise FileNotFoundError(f"Pending resume checkpoint is unavailable: {pending_path}")
    pending_path.replace(resume_path)
    return resume_path


def clear_resume_checkpoint(output_dir: str | Path) -> None:
    """Remove all checkpoint files that can influence a subsequent resume."""

    directory = Path(output_dir)
    for path in (directory / "resume.pending.pt", directory / "resume.pt"):
        path.unlink(missing_ok=True)


def resolve_layer_patterns(model: nn.Module, patterns: Iterable[str]) -> list[str]:
    """Resolve every wildcard pattern in model order, rejecting invalid selections."""

    modules = [(name, module) for name, module in model.named_modules() if name]
    selected_names: set[str] = set()
    for pattern in patterns:
        matches = [(name, module) for name, module in modules if fnmatchcase(name, pattern)]
        if not matches:
            raise ValueError(f"Layer pattern {pattern!r} matched no modules.")
        non_convolutions = [name for name, module in matches if not isinstance(module, nn.Conv2d)]
        if non_convolutions:
            raise ValueError(f"Layer pattern {pattern!r} matched non-Conv2d modules: {non_convolutions!r}.")
        selected_names.update(name for name, _ in matches)
    return [name for name, _ in modules if name in selected_names]


def build_layer_plan(model: nn.Module, layer_name: str) -> LayerPlan:
    """Build one immutable dense-model ranking plan for a convolution layer."""

    modules = dict(model.named_modules())
    module = modules.get(layer_name)
    if not isinstance(module, nn.Conv2d):
        raise ValueError(f"Layer {layer_name!r} must resolve to a torch.nn.Conv2d module.")
    filters_before = int(module.out_channels)
    if filters_before < 2:
        raise ValueError("A planned layer must contain at least two output filters.")
    ranking = tuple(FilterRanking(original_index=index, score=float(score)) for index, score in rank_filters_by_minimum_weight(module))
    return LayerPlan(name=layer_name, filters_before=filters_before, ranking=ranking)


def physical_index(remaining_original_indices: Sequence[int], original_index: int) -> int:
    """Translate a dense-model filter index to its current tensor position."""

    try:
        return remaining_original_indices.index(original_index)
    except ValueError as exc:
        raise ValueError(f"Original filter index {original_index} is not remaining.") from exc


def planned_rows(model_variant: str, hardware: str, layer_plans: Iterable[LayerPlan]) -> list[dict[str, Any]]:
    """Create deterministic, unevaluated rows for every valid pruning step."""

    rows: list[dict[str, Any]] = []
    for plan in layer_plans:
        slug = re.sub(r"[^A-Za-z0-9]+", "-", plan.name).strip("-")
        for filter_rank, entry in enumerate(plan.ranking[:-1], start=1):
            row = {field: None for field in _RESEARCH_FIELDS}
            row.update(
                candidate_id=f"{model_variant}-{hardware}-{slug}-rank-{filter_rank}",
                model_variant=model_variant,
                hardware=hardware,
                status="planned",
                layer=plan.name,
                filter_rank=filter_rank,
                original_filter_index=entry.original_index,
                minimum_weight_score=entry.score,
                filters_before=plan.filters_before,
                filters_removed=filter_rank,
                filters_remaining=plan.filters_before - filter_rank,
            )
            rows.append(row)
    return rows


def complete_candidate_row(
    *,
    layer_plan: LayerPlan,
    ranking_entry: FilterRanking,
    filter_rank: int,
    physical_filter_index: int,
    remaining_original_indices: Sequence[int],
    accuracy: Mapping[str, Any],
    latency: Mapping[str, Any],
    baseline: Mapping[str, Any],
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the completed research row for one measured pruning candidate."""

    model_variant = str(baseline["model_variant"])
    hardware = str(baseline["hardware"])
    slug = re.sub(r"[^A-Za-z0-9]+", "-", layer_plan.name).strip("-")
    row = {
        field: None for field in _RESEARCH_FIELDS
    }
    row.update(
        candidate_id=f"{model_variant}-{hardware}-{slug}-rank-{filter_rank}",
        model_variant=model_variant,
        hardware=hardware,
        stage="single_layer",
        status="completed",
        layer=layer_plan.name,
        filter_rank=filter_rank,
        original_filter_index=ranking_entry.original_index,
        physical_filter_index=physical_filter_index,
        minimum_weight_score=ranking_entry.score,
        filters_before=layer_plan.filters_before,
        filters_removed=filter_rank,
        filters_remaining=len(remaining_original_indices),
        **dict(accuracy),
        **dict(latency),
        **dict(stats),
    )
    if row.get("map50_95") is not None and baseline.get("map50_95") is not None:
        row["map50_95_drop"] = baseline["map50_95"] - row["map50_95"]
    return row


def _load_config(config_path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(config_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Screening configuration must be a mapping.")
    return path, config


def _output_dir(config_path: Path, config: Mapping[str, Any]) -> Path:
    configured = Path(config["experiment"]["output_dir"])
    return configured if configured.is_absolute() else config_path.parent / configured


def _measurement_row(plan: LayerPlan, entry: FilterRanking, rank: int, model_variant: str, hardware: str, status: str, error: BaseException | None = None) -> dict[str, Any]:
    row = next(row for row in planned_rows(model_variant, hardware, [plan]) if row["filter_rank"] == rank)
    row["stage"] = "single_layer"
    row["status"] = status
    if error is not None:
        row["error"] = str(error)
        row["reason"] = "structural" if _is_structural_error(error) else "measurement"
    return row


def _is_structural_error(error: BaseException) -> bool:
    message = str(error).lower()
    return bool(re.search(r"dependency[- ]graph.*\breject", message)) or bool(
        re.search(r"channel(?:s)?\s+(?:mismatch|mis-match)", message)
    )


def _discard_pending_checkpoint(output_dir: Path) -> None:
    (output_dir / "resume.pending.pt").unlink(missing_ok=True)


def run_single_layer_performance_screening(
    config_path: str | Path, *, adapters: ScreeningAdapters | None = None, on_result: Callable[[dict[str, Any]], None] | None = None
) -> list[dict[str, Any]]:
    """Run an independent, complete K-1 pruning curve for each selected layer."""

    if adapters is None:
        raise RuntimeError("Screening adapters must be supplied until runtime adapters are implemented.")
    path, config = _load_config(config_path)
    output_dir = _output_dir(path, config)
    checkpoint = Path(config["model"]["checkpoint"])
    dataset = Path(config["data"]["dataset_yaml"])
    fingerprint = experiment_fingerprint(config, checkpoint, dataset)
    experiment = config["experiment"]
    model_variant = str(experiment["model_variant"])
    hardware = str(experiment["hardware_label"])

    planning_wrapper = adapters.load_model(checkpoint)
    planning_model = getattr(planning_wrapper, "model", planning_wrapper)
    layers = resolve_layer_patterns(planning_model, config["screening"]["layer_patterns"])
    layer_plans = [build_layer_plan(planning_model, layer) for layer in layers]
    rows, _saved_rankings, state = load_screening_artifacts(output_dir, fingerprint)
    rows_by_id = {str(row["candidate_id"]): dict(row) for row in rows}
    rankings = {
        plan.name: [{"original_index": entry.original_index, "score": entry.score} for entry in plan.ranking]
        for plan in layer_plans
    }
    for planned in planned_rows(model_variant, hardware, layer_plans):
        planned["stage"] = "single_layer"
        rows_by_id.setdefault(planned["candidate_id"], planned)

    baseline_row = rows_by_id.get("baseline")
    if baseline_row is None:
        baseline_accuracy = dict(adapters.evaluate(planning_wrapper, config))
        baseline_latency = dict(adapters.profile(planning_wrapper, config))
        baseline_row = {
            "candidate_id": "baseline", "stage": "baseline", "status": "completed", "model_variant": model_variant,
            "hardware": hardware, **baseline_accuracy, **baseline_latency, **dict(adapters.stats(planning_wrapper)),
        }
        rows_by_id["baseline"] = baseline_row
        write_screening_artifacts(output_dir, path, list(rows_by_id.values()), rankings, state)
        if on_result:
            on_result(baseline_row)

    for plan in layer_plans:
        if state is not None and state.active_layer != plan.name:
            continue
        resuming = state is not None and state.active_layer == plan.name
        if resuming:
            remaining = list(state.remaining_original_indices)
            next_rank = state.next_filter_rank
            if next_rank == 1:
                working_model = adapters.load_model(checkpoint)
            else:
                resume_checkpoint = output_dir / "resume.pt"
                if not resume_checkpoint.is_file():
                    raise FileNotFoundError(f"Resume checkpoint is required for rank {next_rank}: {resume_checkpoint}")
                working_model = adapters.load_model(resume_checkpoint)
        else:
            working_model = adapters.load_model(checkpoint)
            remaining = list(range(plan.filters_before))
            next_rank = 1

        for rank, entry in enumerate(plan.ranking[:-1], start=1):
            if rank < next_rank:
                continue
            physical = physical_index(remaining, entry.original_index)
            try:
                working_model = adapters.prune_filter(working_model, plan.name, physical, config)
                save_pending_checkpoint(working_model, output_dir, adapters.save_checkpoint)
                evaluation_model = adapters.load_model(output_dir / "resume.pending.pt")
                accuracy = adapters.evaluate(evaluation_model, config)
                latency = adapters.profile(working_model, config)
                next_remaining = list(remaining)
                next_remaining.remove(entry.original_index)
                completed = complete_candidate_row(
                    layer_plan=plan, ranking_entry=entry, filter_rank=rank, physical_filter_index=physical,
                    remaining_original_indices=next_remaining, accuracy=accuracy, latency=latency, baseline=baseline_row,
                    stats=adapters.stats(working_model),
                )
            except BaseException as error:
                _discard_pending_checkpoint(output_dir)
                if _is_structural_error(error):
                    for skipped_rank, skipped_entry in enumerate(plan.ranking[rank - 1:-1], start=rank):
                        skipped = _measurement_row(plan, skipped_entry, skipped_rank, model_variant, hardware, "skipped", error)
                        rows_by_id[skipped["candidate_id"]] = skipped
                    write_screening_artifacts(output_dir, path, list(rows_by_id.values()), rankings, state)
                    if on_result:
                        for skipped_rank, skipped_entry in enumerate(plan.ranking[rank - 1:-1], start=rank):
                            candidate_id = _measurement_row(plan, skipped_entry, skipped_rank, model_variant, hardware, "skipped")["candidate_id"]
                            on_result(rows_by_id[candidate_id])
                    break
                failed = _measurement_row(plan, entry, rank, model_variant, hardware, "failed", error)
                rows_by_id[failed["candidate_id"]] = failed
                if state is None:
                    state = ResumeState(1, fingerprint, plan.name, rank, tuple(remaining), None)
                write_screening_artifacts(output_dir, path, list(rows_by_id.values()), rankings, state)
                if on_result:
                    on_result(failed)
                raise
            remaining = next_remaining
            promote_pending_checkpoint(output_dir)
            rows_by_id[completed["candidate_id"]] = completed
            state = ResumeState(1, fingerprint, plan.name, rank + 1, tuple(remaining), completed["candidate_id"])
            write_screening_artifacts(output_dir, path, list(rows_by_id.values()), rankings, state)
            if on_result:
                on_result(completed)
        clear_resume_checkpoint(output_dir)
        state = None
        write_screening_artifacts(output_dir, path, list(rows_by_id.values()), rankings, state)

    return list(rows_by_id.values())
