from __future__ import annotations

import csv
import copy
import json
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

from infrared_detection.evaluation.single_layer_performance_screening import (
    FilterRanking,
    LayerPlan,
    ResumeState,
    ScreeningAdapters,
    build_layer_plan,
    experiment_fingerprint,
    load_screening_artifacts,
    physical_index,
    planned_rows,
    promote_pending_checkpoint,
    resolve_layer_patterns,
    save_pending_checkpoint,
    run_single_layer_performance_screening,
    write_screening_artifacts,
)
import infrared_detection.evaluation.single_layer_performance_screening as screening_module


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity(),
            nn.ModuleDict({
                "m": nn.ModuleList([
                    nn.ModuleDict({"cv1": nn.ModuleDict({"conv": nn.Conv2d(1, 4, 1)}), "cv2": nn.ModuleDict({"conv": nn.Conv2d(1, 4, 1)})}),
                    nn.ModuleDict({"cv1": nn.ModuleDict({"conv": nn.Conv2d(1, 4, 1)}), "cv2": nn.ModuleDict({"conv": nn.Conv2d(1, 4, 1)})}),
                ])
            }),
        )


@pytest.fixture
def model() -> TinyModel:
    torch.manual_seed(7)
    return TinyModel()


def test_patterns_resolve_in_model_order_and_deduplicate_matches(model: TinyModel):
    assert resolve_layer_patterns(model, ["model.6.m.*.cv1.conv", "model.6.m.*.cv1.conv"]) == [
        "model.6.m.0.cv1.conv",
        "model.6.m.1.cv1.conv",
    ]


def test_patterns_reject_unmatched_patterns(model: TinyModel):
    with pytest.raises(ValueError, match="matched no modules"):
        resolve_layer_patterns(model, ["model.99.*"])


def test_patterns_reject_non_convolution_matches(model: TinyModel):
    with pytest.raises(ValueError, match="Conv2d"):
        resolve_layer_patterns(model, ["model.6.m.0.cv1"])


def test_plan_contains_immutable_ranking_and_three_candidate_rows(model: TinyModel):
    plan = build_layer_plan(model, "model.6.m.0.cv1.conv")

    assert isinstance(plan, LayerPlan)
    assert isinstance(plan.ranking[0], FilterRanking)
    assert plan.filters_before == 4
    assert len(planned_rows("yolov8n", "rtx3070", [plan])) == 3
    with pytest.raises(AttributeError):
        plan.filters_before = 3


def test_plan_rejects_layers_with_fewer_than_two_filters(model: TinyModel):
    model.model[6].m[0].cv1.conv = nn.Conv2d(1, 1, 1)

    with pytest.raises(ValueError, match="at least two"):
        build_layer_plan(model, "model.6.m.0.cv1.conv")


def test_physical_index_translates_from_remaining_original_indices():
    assert physical_index([0, 2, 3], 2) == 1


def test_planned_rows_include_research_fields_and_deterministic_candidate_ids():
    plan = LayerPlan(
        name="model.6.m.0.cv1.conv",
        filters_before=3,
        ranking=(FilterRanking(2, 0.1), FilterRanking(0, 0.2), FilterRanking(1, 0.3)),
    )

    rows = planned_rows("yolov8n", "rtx3070", [plan])

    assert [row["candidate_id"] for row in rows] == [
        "yolov8n-rtx3070-model-6-m-0-cv1-conv-rank-1",
        "yolov8n-rtx3070-model-6-m-0-cv1-conv-rank-2",
    ]
    assert rows[0]["filters_remaining"] == 2
    assert rows[1]["filters_remaining"] == 1
    assert rows[0]["status"] == "planned"
    assert set((
        "candidate_id", "model_variant", "hardware", "status", "layer", "filter_rank",
        "original_filter_index", "physical_filter_index", "minimum_weight_score", "filters_before",
        "filters_removed", "filters_remaining", "map50_95", "map50_95_drop", "map50", "precision",
        "recall", "latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "fps",
    )) <= rows[0].keys()


def test_artifacts_write_ordered_csv_and_reject_mismatched_fingerprint(tmp_path):
    output_dir = tmp_path / "artifacts"
    config_path = tmp_path / "screening.yaml"
    config_path.write_text("experiment: {}\n", encoding="utf-8")
    state = ResumeState(
        version=1,
        fingerprint="abc123",
        active_layer="model.6.m.0.cv1.conv",
        next_filter_rank=2,
        remaining_original_indices=(0, 2, 3),
        last_candidate_id="yolov8n-rtx3070-model-6-m-0-cv1-conv-rank-1",
    )

    write_screening_artifacts(
        output_dir,
        config_path,
        rows=[{"candidate_id": "baseline", "status": "completed"}],
        rankings={"model.6.m.0.cv1.conv": [{"original_index": 1, "score": 0.01}]},
        state=state,
    )

    assert (output_dir / "results.csv").exists()
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "rankings.json").exists()
    assert (output_dir / "state.json").exists()
    with (output_dir / "results.csv").open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert header[:21] == [
        "candidate_id", "model_variant", "hardware", "status", "layer", "filter_rank",
        "original_filter_index", "physical_filter_index", "minimum_weight_score", "filters_before",
        "filters_removed", "filters_remaining", "map50_95", "map50_95_drop", "map50", "precision",
        "recall", "latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "fps",
    ]
    assert header[-2:] == ["reason", "error"]
    assert json.loads((output_dir / "state.json").read_text(encoding="utf-8"))["remaining_original_indices"] == [0, 2, 3]

    with pytest.raises(ValueError, match="experiment fingerprint"):
        load_screening_artifacts(output_dir, "different-fingerprint")


def test_relative_output_directory_is_resolved_from_the_invocation_directory(monkeypatch, tmp_path):
    config_path = tmp_path / "configs" / "experiments" / "screening.yaml"
    config_path.parent.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    output = screening_module._output_dir(
        config_path,
        {"experiment": {"output_dir": "runs/experiments/single_layer/yolov8n/rtx3070"}},
    )

    assert output == tmp_path / "runs" / "experiments" / "single_layer" / "yolov8n" / "rtx3070"


def test_experiment_fingerprint_uses_canonical_config_and_input_file_metadata(tmp_path):
    checkpoint = tmp_path / "model.pt"
    dataset = tmp_path / "dataset.yaml"
    checkpoint.write_bytes(b"checkpoint")
    dataset.write_text("path: data\n", encoding="utf-8")
    config = {
        "experiment": {"model_variant": "yolov8n", "hardware_label": "rtx3070", "image_size": 640},
        "runtime": {"precision": "fp16"},
        "screening": {"layer_patterns": ["model.*.conv"]},
    }

    fingerprint = experiment_fingerprint(config, checkpoint, dataset)

    assert fingerprint == experiment_fingerprint(dict(config), checkpoint, dataset)
    assert len(fingerprint) == 64
    dataset.write_text("path: changed\n", encoding="utf-8")
    assert fingerprint != experiment_fingerprint(config, checkpoint, dataset)


class SaveCapableModel:
    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(b"pruned checkpoint")


def test_resume_checkpoint_promotes_pending_file_atomically(tmp_path):
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    (output_dir / "resume.pt").write_bytes(b"previous checkpoint")

    pending = save_pending_checkpoint(SaveCapableModel(), output_dir)

    assert pending == output_dir / "resume.pending.pt"
    assert pending.is_file() and pending.stat().st_size > 0
    promoted = promote_pending_checkpoint(output_dir)
    assert promoted == output_dir / "resume.pt"
    assert promoted.read_bytes() == b"pruned checkpoint"
    assert not pending.exists()


_PROJECT_ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.skipif(not (_PROJECT_ROOT / "yolov8n.pt").exists(), reason="yolov8n.pt is unavailable")
def test_supplied_yolov8n_architecture_has_expected_layer_and_candidate_counts():
    from ultralytics import YOLO

    model = YOLO(str(_PROJECT_ROOT / "yolov8n.pt")).model
    layers = resolve_layer_patterns(model, _screening_patterns())
    plans = [build_layer_plan(model, layer) for layer in layers]

    assert (len(layers), sum(plan.filters_before - 1 for plan in plans)) == (12, 1012)


@pytest.mark.skipif(not (_PROJECT_ROOT / "yolov8m.pt").exists(), reason="yolov8m.pt is unavailable")
def test_supplied_yolov8m_architecture_has_expected_layer_and_candidate_counts():
    from ultralytics import YOLO

    model = YOLO(str(_PROJECT_ROOT / "yolov8m.pt")).model
    layers = resolve_layer_patterns(model, _screening_patterns())
    plans = [build_layer_plan(model, layer) for layer in layers]

    assert (len(layers), sum(plan.filters_before - 1 for plan in plans)) == (24, 5352)


def _screening_patterns() -> list[str]:
    return [
        "model.6.m.*.cv1.conv", "model.6.m.*.cv2.conv", "model.8.m.*.cv1.conv", "model.8.m.*.cv2.conv",
        "model.12.m.*.cv1.conv", "model.12.m.*.cv2.conv", "model.18.m.*.cv1.conv", "model.18.m.*.cv2.conv",
        "model.21.m.*.cv1.conv", "model.21.m.*.cv2.conv",
    ]


class ScreeningModel(nn.Module):
    def __init__(self, history: tuple[tuple[str, int], ...] = ()) -> None:
        super().__init__()
        self.layer_a = nn.Conv2d(1, 4, 1, bias=False)
        self.layer_b = nn.Conv2d(1, 4, 1, bias=False)
        with torch.no_grad():
            for layer in (self.layer_a, self.layer_b):
                layer.weight[:, 0, 0, 0] = torch.tensor([2.0, 4.0, 1.0, 3.0])
        self.history = history

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.history), encoding="utf-8")


def _screening_config(tmp_path: Path, layers: list[str]) -> Path:
    checkpoint = tmp_path / "dense.pt"
    dataset = tmp_path / "dataset.yaml"
    checkpoint.write_bytes(b"dense")
    dataset.write_text("path: data\n", encoding="utf-8")
    config_path = tmp_path / "screening.yaml"
    config_path.write_text(
        "\n".join((
            "model:", f"  checkpoint: {checkpoint}", "data:", f"  dataset_yaml: {dataset}",
            "experiment:", "  model_variant: fake", "  hardware_label: cpu", "  image_size: 32",
            f"  output_dir: {tmp_path / 'artifacts'}", "runtime:", "  precision: fp32", "screening:",
            "  layer_patterns:", *(f"    - {layer}" for layer in layers), "",
        )),
        encoding="utf-8",
    )
    return config_path


def _screening_adapters(*, profile=None, prune=None, stats=None):
    loads: list[tuple[Path, tuple[tuple[str, int], ...]]] = []
    evaluations: list[tuple[tuple[str, int], ...]] = []
    prunes: list[tuple[str, int, tuple[tuple[str, int], ...]]] = []

    def load_model(path: str | Path) -> ScreeningModel:
        checkpoint = Path(path)
        history = () if checkpoint.name == "dense.pt" else tuple(tuple(item) for item in json.loads(checkpoint.read_text(encoding="utf-8")))
        model = ScreeningModel(history)
        loads.append((checkpoint, model.history))
        return model

    def prune_filter(model: ScreeningModel, layer: str, physical_index: int, config: dict) -> ScreeningModel:
        prunes.append((layer, physical_index, model.history))
        if prune is not None:
            prune(layer, physical_index, model)
        pruned = copy.deepcopy(model)
        pruned.history = (*model.history, (layer, physical_index))
        return pruned

    def evaluate(model: ScreeningModel, config: dict) -> dict:
        evaluations.append(model.history)
        return {"map50_95": 0.5, "map50": 0.6, "precision": 0.7, "recall": 0.8}

    def profile_model(model: ScreeningModel, config: dict) -> dict:
        if profile is not None:
            profile(model)
        return {"latency_mean_ms": 10.0, "latency_p50_ms": 9.0, "latency_p95_ms": 12.0, "fps": 100.0}

    adapters = ScreeningAdapters(
        load_model=load_model,
        evaluate=evaluate,
        prune_filter=prune_filter,
        profile=profile_model,
        stats=lambda model: {} if stats is None else stats(model),
        save_checkpoint=lambda model, path: model.save(path),
    )
    return adapters, loads, evaluations, prunes


def test_complete_curve_records_k_minus_one_candidates_and_translated_physical_indices(tmp_path):
    config_path = _screening_config(tmp_path, ["layer_a"])
    adapters, _loads, _evaluations, prunes = _screening_adapters()

    rows = run_single_layer_performance_screening(config_path, adapters=adapters)

    candidates = [row for row in rows if row["stage"] == "single_layer"]
    assert [row["filter_rank"] for row in candidates] == [1, 2, 3]
    assert [row["filters_remaining"] for row in candidates] == [3, 2, 1]
    assert [row["status"] for row in candidates] == ["completed"] * 3
    assert [physical for _layer, physical, _history in prunes] == [2, 0, 1]
    assert not (tmp_path / "artifacts" / "resume.pt").exists()


def test_planning_resolves_configured_paths_against_a_wrapped_model(tmp_path):
    config_path = _screening_config(tmp_path, ["layer_a"])

    class Wrapper(nn.Module):
        def __init__(self, inner: ScreeningModel | None = None) -> None:
            super().__init__()
            self.model = inner or ScreeningModel()

    def load_model(path: str | Path) -> Wrapper:
        checkpoint = Path(path)
        history = () if checkpoint.name == "dense.pt" else tuple(
            tuple(item) for item in json.loads(checkpoint.read_text(encoding="utf-8"))
        )
        return Wrapper(ScreeningModel(history))

    def prune_filter(wrapper: Wrapper, layer: str, physical_index: int, config: dict) -> Wrapper:
        pruned = copy.deepcopy(wrapper)
        pruned.model.history = (*wrapper.model.history, (layer, physical_index))
        return pruned

    adapters = ScreeningAdapters(
        load_model=load_model,
        evaluate=lambda wrapper, config: {"map50_95": 0.5},
        prune_filter=prune_filter,
        profile=lambda wrapper, config: {"latency_p50_ms": 9.0},
        stats=lambda wrapper: {},
        save_checkpoint=lambda wrapper, path: Path(path).write_text(
            json.dumps(wrapper.model.history), encoding="utf-8"
        ),
    )

    rows = run_single_layer_performance_screening(config_path, adapters=adapters)

    assert [row["status"] for row in rows if row.get("stage") == "single_layer"] == ["completed"] * 3


def test_independent_layers_each_start_from_the_dense_checkpoint(tmp_path):
    config_path = _screening_config(tmp_path, ["layer_a", "layer_b"])
    adapters, loads, _evaluations, prunes = _screening_adapters()

    run_single_layer_performance_screening(config_path, adapters=adapters)

    dense_loads = [history for checkpoint, history in loads if checkpoint.name == "dense.pt"]
    assert dense_loads == [(), (), ()]
    first_layer_b_prune = next(history for layer, _physical, history in prunes if layer == "layer_b")
    assert first_layer_b_prune == ()


def test_resume_does_not_reevaluate_completed_ranks(tmp_path):
    config_path = _screening_config(tmp_path, ["layer_a"])
    profile_calls = 0

    def interrupt_after_rank_two(model):
        nonlocal profile_calls
        profile_calls += 1
        if profile_calls == 4:  # baseline, ranks 1 and 2, then rank 3
            raise RuntimeError("interrupted")

    interrupted_adapters, _loads, first_evaluations, _prunes = _screening_adapters(profile=interrupt_after_rank_two)
    with pytest.raises(RuntimeError, match="interrupted"):
        run_single_layer_performance_screening(config_path, adapters=interrupted_adapters)

    rows, _rankings, _state = load_screening_artifacts(tmp_path / "artifacts", _artifact_fingerprint(config_path))
    assert [row["filter_rank"] for row in rows if row.get("stage") == "single_layer" and row.get("status") == "completed"] == [1, 2]
    assert (tmp_path / "artifacts" / "resume.pt").exists()

    resumed_adapters, _loads, resumed_evaluations, _prunes = _screening_adapters()
    rows = run_single_layer_performance_screening(config_path, adapters=resumed_adapters)

    assert len(first_evaluations) == 4  # baseline plus ranks 1, 2, and 3 before profiling interrupts
    assert resumed_evaluations == [(('layer_a', 2), ('layer_a', 0), ('layer_a', 1))]
    assert [row["filter_rank"] for row in rows if row.get("stage") == "single_layer" and row.get("status") == "completed"] == [1, 2, 3]


def test_structural_failure_skips_the_rest_of_its_layer_and_continues_with_the_next(tmp_path):
    config_path = _screening_config(tmp_path, ["layer_a", "layer_b"])

    def reject_second_layer_a_prune(layer, physical_index, model):
        if layer == "layer_a" and physical_index == 0:
            raise RuntimeError("channel mismatch in dependency graph")

    adapters, _loads, _evaluations, prunes = _screening_adapters(prune=reject_second_layer_a_prune)
    rows = run_single_layer_performance_screening(config_path, adapters=adapters)

    layer_a = [row for row in rows if row.get("layer") == "layer_a"]
    assert [row["status"] for row in layer_a] == ["completed", "skipped", "skipped"]
    assert all(row["status"] == "completed" for row in rows if row.get("layer") == "layer_b")
    assert any(layer == "layer_b" for layer, _physical, _history in prunes)


def test_out_of_memory_persists_failed_rank_stops_then_retries_it_on_resume(tmp_path):
    config_path = _screening_config(tmp_path, ["layer_a"])

    def out_of_memory(model):
        if model.history == (("layer_a", 2),):
            raise torch.OutOfMemoryError("CUDA out of memory")

    failing_adapters, _loads, _evaluations, _prunes = _screening_adapters(profile=out_of_memory)
    with pytest.raises(torch.OutOfMemoryError, match="out of memory"):
        run_single_layer_performance_screening(config_path, adapters=failing_adapters)

    rows, _rankings, state = load_screening_artifacts(tmp_path / "artifacts", _artifact_fingerprint(config_path))
    candidates = [row for row in rows if row.get("stage") == "single_layer"]
    assert [row["status"] for row in candidates] == ["failed", "planned", "planned"]
    assert state is not None and state.next_filter_rank == 1

    retry_adapters, _loads, retry_evaluations, _prunes = _screening_adapters()
    rows = run_single_layer_performance_screening(config_path, adapters=retry_adapters)
    assert retry_evaluations[0] == (("layer_a", 2),)
    assert [row["status"] for row in rows if row.get("stage") == "single_layer"] == ["completed"] * 3


def test_failed_measurement_removes_only_pending_checkpoint_and_preserves_prior_resume(tmp_path):
    config_path = _screening_config(tmp_path, ["layer_a"])

    def fail_rank_two_measurement(model):
        if model.history == (("layer_a", 2), ("layer_a", 0)):
            raise RuntimeError("measurement failed")

    adapters, _loads, _evaluations, _prunes = _screening_adapters(profile=fail_rank_two_measurement)
    with pytest.raises(RuntimeError, match="measurement failed"):
        run_single_layer_performance_screening(config_path, adapters=adapters)

    output_dir = tmp_path / "artifacts"
    assert not (output_dir / "resume.pending.pt").exists()
    assert json.loads((output_dir / "resume.pt").read_text(encoding="utf-8")) == [["layer_a", 2]]


def test_stats_failure_preserves_previous_resume_and_marks_the_rank_retryable(tmp_path):
    config_path = _screening_config(tmp_path, ["layer_a"])

    def fail_second_rank_stats(model):
        if model.history == (("layer_a", 2), ("layer_a", 0)):
            raise RuntimeError("stats failed")
        return {}

    adapters, _loads, _evaluations, _prunes = _screening_adapters(stats=fail_second_rank_stats)
    with pytest.raises(RuntimeError, match="stats failed"):
        run_single_layer_performance_screening(config_path, adapters=adapters)

    output_dir = tmp_path / "artifacts"
    rows, _rankings, state = load_screening_artifacts(output_dir, _artifact_fingerprint(config_path))
    candidates = [row for row in rows if row.get("stage") == "single_layer"]
    assert [row["status"] for row in candidates] == ["completed", "failed", "planned"]
    assert state is not None and state.next_filter_rank == 2
    assert json.loads((output_dir / "resume.pt").read_text(encoding="utf-8")) == [["layer_a", 2]]
    assert not (output_dir / "resume.pending.pt").exists()


def test_dependency_service_error_is_failed_retryable_and_rethrown_not_structural(tmp_path):
    config_path = _screening_config(tmp_path, ["layer_a"])

    def fail_with_transient_dependency_error(layer, physical_index, model):
        raise RuntimeError("dependency service unavailable")

    adapters, _loads, _evaluations, _prunes = _screening_adapters(prune=fail_with_transient_dependency_error)
    with pytest.raises(RuntimeError, match="dependency service unavailable"):
        run_single_layer_performance_screening(config_path, adapters=adapters)

    rows, _rankings, state = load_screening_artifacts(tmp_path / "artifacts", _artifact_fingerprint(config_path))
    candidates = [row for row in rows if row.get("stage") == "single_layer"]
    assert [row["status"] for row in candidates] == ["failed", "planned", "planned"]
    assert state is not None and state.next_filter_rank == 1


class NoSaveScreeningModel(nn.Module):
    def __init__(self, history: tuple[tuple[str, int], ...] = ()) -> None:
        super().__init__()
        self.layer_a = nn.Conv2d(1, 4, 1, bias=False)
        with torch.no_grad():
            self.layer_a.weight[:, 0, 0, 0] = torch.tensor([2.0, 4.0, 1.0, 3.0])
        self.history = history


def test_checkpoint_adapter_saves_pending_checkpoint_when_working_model_has_no_save(tmp_path):
    config_path = _screening_config(tmp_path, ["layer_a"])
    saved_paths: list[Path] = []

    def load_model(path: str | Path) -> NoSaveScreeningModel:
        checkpoint = Path(path)
        history = () if checkpoint.name == "dense.pt" else tuple(tuple(item) for item in json.loads(checkpoint.read_text(encoding="utf-8")))
        return NoSaveScreeningModel(history)

    def prune_filter(model, layer, physical_index, config):
        pruned = copy.deepcopy(model)
        pruned.history = (*model.history, (layer, physical_index))
        return pruned

    def save_checkpoint(model, path):
        target = Path(path)
        saved_paths.append(target)
        target.write_text(json.dumps(model.history), encoding="utf-8")

    adapters = ScreeningAdapters(
        load_model=load_model,
        evaluate=lambda model, config: {"map50_95": 0.5},
        prune_filter=prune_filter,
        profile=lambda model, config: {"latency_mean_ms": 10.0},
        stats=lambda model: {},
        save_checkpoint=save_checkpoint,
    )

    run_single_layer_performance_screening(config_path, adapters=adapters)

    assert saved_paths == [tmp_path / "artifacts" / "resume.pending.pt"] * 3


def _artifact_fingerprint(config_path: Path) -> str:
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return experiment_fingerprint(config, config["model"]["checkpoint"], config["data"]["dataset_yaml"])


class _ProductionWrapper:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.val_calls: list[dict] = []

    def val(self, **kwargs):
        self.val_calls.append(kwargs)
        return types.SimpleNamespace(
            box=types.SimpleNamespace(
                map=0.75408,
                map50=0.901,
                mp=0.812,
                mr=0.723,
                maps=[0.71, 0.72, 0.73, 0.74],
            )
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(b"wrapped checkpoint")


class _ProfileModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Conv2d(3, 4, 1)
        self.moves: list[object] = []
        self.half_calls = 0

    def to(self, *args, **kwargs):
        self.moves.append(args[0] if args else kwargs.get("device"))
        return self

    def half(self):
        self.half_calls += 1
        return self


def _production_config() -> dict:
    return {
        "model": {"checkpoint": "models/checkpoints/yolov8/train3/weights/best.pt"},
        "data": {"dataset_yaml": "data/camel/camel.yaml", "split": "val"},
        "experiment": {"image_size": 352, "batch_size": 1},
        "runtime": {"device": "0", "precision": "fp16", "conf": 0.25, "iou": 0.6},
        "screening": {"warmup": 20, "iterations": 100},
    }


def test_production_adapters_evaluate_with_yolo_validation_arguments_and_normalise_metrics(monkeypatch):
    wrapper = _ProductionWrapper(nn.Conv2d(3, 4, 1))
    yolo_paths: list[str] = []
    stats_calls: list[nn.Module] = []

    class FakeYOLO:
        def __new__(cls, checkpoint: str):
            yolo_paths.append(checkpoint)
            return wrapper

    def fake_collect_model_stats(model):
        stats_calls.append(model)
        return {"parameters": 3_011_628}

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    monkeypatch.setattr(screening_module, "collect_model_stats", fake_collect_model_stats, raising=False)

    adapters = ScreeningAdapters.defaults(_production_config())
    loaded = adapters.load_model("models/checkpoints/yolov8/train3/weights/best.pt")
    metrics = adapters.evaluate(loaded, _production_config())
    stats = adapters.stats(loaded)

    assert yolo_paths == ["models/checkpoints/yolov8/train3/weights/best.pt"]
    assert wrapper.val_calls == [{
        "data": "data/camel/camel.yaml", "split": "val", "imgsz": 352, "device": "0",
        "conf": 0.25, "iou": 0.6, "batch": 1, "half": True, "verbose": False,
    }]
    assert metrics == {
        "map50_95": 0.75408, "map50": 0.901, "precision": 0.812, "recall": 0.723,
        "per_class_ap": [0.71, 0.72, 0.73, 0.74],
    }
    assert stats == {"parameters": 3_011_628}
    assert stats_calls == [wrapper.model]


def test_production_dependencies_resolve_from_their_authoritative_modules():
    assert screening_module._production_dependency("run_filterwise_probe").__module__ == (
        "infrared_detection.compression.pruning.cluster_probe"
    )
    assert screening_module._production_dependency("benchmark_pytorch_cuda_forward").__module__ == (
        "infrared_detection.benchmarking.pytorch_cuda"
    )
    assert screening_module._production_dependency("collect_model_stats").__module__ == (
        "infrared_detection.evaluation.model_stats"
    )


def test_linux_checkpoint_compatibility_maps_windows_path_pickles(monkeypatch):
    import pathlib

    monkeypatch.setattr(screening_module.sys, "platform", "linux")
    monkeypatch.delitem(screening_module.sys.modules, "pathlib._local", raising=False)

    screening_module._install_legacy_pathlib_checkpoint_compatibility()

    legacy_module = screening_module.sys.modules["pathlib._local"]
    assert legacy_module.WindowsPath is pathlib.PosixPath


def test_production_adapters_prune_one_physical_index_with_a_352_pixel_example_input(monkeypatch):
    wrapper = _ProductionWrapper(nn.Conv2d(3, 4, 1))
    original_model = wrapper.model
    probe_calls: list[tuple[nn.Module, torch.Tensor, str, tuple[int, ...]]] = []

    def fake_probe(model, example_input, layer, physical_indices):
        probe_calls.append((model, example_input, layer, physical_indices))
        return model

    monkeypatch.setattr(screening_module, "run_filterwise_probe", fake_probe, raising=False)
    adapters = ScreeningAdapters.defaults(_production_config())

    pruned = adapters.prune_filter(wrapper, "model.6.m.0.cv1.conv", 17, _production_config())

    assert pruned is wrapper
    assert pruned.model is original_model
    model, example_input, layer, physical_indices = probe_calls[0]
    assert model is wrapper.model
    assert tuple(example_input.shape) == (1, 3, 352, 352)
    assert layer == "model.6.m.0.cv1.conv"
    assert physical_indices == (17,)


def test_production_pruning_trace_matches_the_current_model_device_and_dtype(monkeypatch):
    model = nn.Conv2d(3, 4, 1, device="meta", dtype=torch.float64)
    wrapper = _ProductionWrapper(model)
    trace_inputs: list[torch.Tensor] = []

    def fake_probe(pruned_model, example_input, layer, physical_indices):
        trace_inputs.append(example_input)
        return pruned_model

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(screening_module, "run_filterwise_probe", fake_probe, raising=False)
    adapters = ScreeningAdapters.defaults(_production_config())

    adapters.prune_filter(wrapper, "model.6.m.0.cv1.conv", 17, _production_config())

    assert trace_inputs[0].device == model.weight.device
    assert trace_inputs[0].dtype == model.weight.dtype


def test_production_adapters_profile_on_configured_cuda_device_in_requested_precision(monkeypatch):
    model = _ProfileModel()
    wrapper = _ProductionWrapper(model)
    benchmark_calls: list[tuple[nn.Module, torch.Tensor, dict]] = []
    tensor_moves: list[object] = []
    original_tensor_to = torch.Tensor.to

    def fake_tensor_to(tensor, *args, **kwargs):
        target = args[0] if args else kwargs.get("device")
        tensor_moves.append(target)
        if target in ("cuda:0", torch.device("cuda:0")):
            return tensor
        return original_tensor_to(tensor, *args, **kwargs)

    def fake_benchmark(profiled_model, example_input, **kwargs):
        benchmark_calls.append((profiled_model, example_input, kwargs))
        return {"latency_mean_ms": 3.5, "latency_p50_ms": 3.2, "latency_p95_ms": 4.1, "fps": 285.7}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.Tensor, "to", fake_tensor_to)
    monkeypatch.setattr(screening_module, "benchmark_pytorch_cuda_forward", fake_benchmark, raising=False)
    adapters = ScreeningAdapters.defaults(_production_config())

    profile = adapters.profile(wrapper, _production_config())

    assert model.moves == ["cuda:0"]
    assert model.half_calls == 1
    assert tensor_moves == ["cuda:0"]
    profiled_model, example_input, kwargs = benchmark_calls[0]
    assert profiled_model is model
    assert tuple(example_input.shape) == (1, 3, 352, 352)
    assert example_input.dtype == torch.float16
    assert kwargs == {"warmup": 20, "iterations": 100}
    assert profile["latency_p50_ms"] == 3.2


def test_production_adapters_fail_fast_when_cuda_is_unavailable_before_baseline_evaluation(monkeypatch):
    baseline_evaluations = 0

    def fake_yolo(_checkpoint):
        nonlocal baseline_evaluations
        wrapper = _ProductionWrapper(nn.Conv2d(3, 4, 1))
        original_val = wrapper.val

        def val(**kwargs):
            nonlocal baseline_evaluations
            baseline_evaluations += 1
            return original_val(**kwargs)

        wrapper.val = val
        return wrapper

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=fake_yolo))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA.*unavailable"):
        ScreeningAdapters.defaults(_production_config())

    assert baseline_evaluations == 0
