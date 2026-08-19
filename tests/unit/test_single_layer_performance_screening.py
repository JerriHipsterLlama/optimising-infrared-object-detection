from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from infrared_detection.evaluation.single_layer_performance_screening import (
    FilterRanking,
    LayerPlan,
    build_layer_plan,
    physical_index,
    planned_rows,
    resolve_layer_patterns,
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
