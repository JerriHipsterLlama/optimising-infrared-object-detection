from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from infrared_detection.evaluation.single_layer_performance_screening import (
    FilterRanking,
    LayerPlan,
    ResumeState,
    build_layer_plan,
    experiment_fingerprint,
    load_screening_artifacts,
    physical_index,
    planned_rows,
    promote_pending_checkpoint,
    resolve_layer_patterns,
    save_pending_checkpoint,
    write_screening_artifacts,
)


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
