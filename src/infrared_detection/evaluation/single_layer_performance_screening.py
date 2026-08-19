"""Deterministic planning primitives for single-layer performance screening."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from torch import nn

from infrared_detection.compression.pruning import rank_filters_by_minimum_weight
from infrared_detection.evaluation.artifacts import write_metrics_csv


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


def save_pending_checkpoint(model: Any, output_dir: str | Path) -> Path:
    """Save a candidate checkpoint without disturbing the last valid resume point."""

    pending_path = Path(output_dir) / "resume.pending.pt"
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    save = getattr(model, "save", None)
    if not callable(save):
        raise RuntimeError("Resume checkpoint requires a save-capable model.")
    save(pending_path)
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
