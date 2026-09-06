from __future__ import annotations

import pytest

from infrared_detection.evaluation.accuracy_sensitivity import (
    DEFAULT_RATIOS,
    build_unit_rankings,
    calculate_point_metrics,
    normalized_degradation_auc,
)


@pytest.mark.parametrize(
    ("baseline", "candidate", "ratio", "degradation", "change", "sensitivity"),
    [
        (0.50, 0.45, 0.25, 0.05, -0.05, 0.20),
        (0.50, 0.52, 0.25, -0.02, 0.02, -0.08),
    ],
)
def test_point_metrics_preserve_signed_accuracy_change(
    baseline, candidate, ratio, degradation, change, sensitivity
):
    result = calculate_point_metrics(baseline, candidate, ratio)

    assert result.accuracy_degradation == pytest.approx(degradation)
    assert result.map50_95_change_from_baseline == pytest.approx(change)
    assert result.point_sensitivity == pytest.approx(sensitivity)


def test_point_metrics_reject_zero_achieved_ratio():
    with pytest.raises(ValueError, match="positive"):
        calculate_point_metrics(0.5, 0.4, 0.0)


def test_normalized_auc_includes_zero_origin_and_uses_achieved_ratios():
    points = [(0.125, 0.01), (0.25, 0.03), (0.375, 0.06), (0.5, 0.10)]

    assert normalized_degradation_auc(points) == pytest.approx(0.0375)


def test_normalized_auc_preserves_negative_accuracy_degradation():
    assert normalized_degradation_auc([(0.25, -0.02), (0.5, -0.04)]) == pytest.approx(-0.02)


def test_normalized_auc_rejects_duplicate_achieved_ratios():
    with pytest.raises(ValueError, match="strictly increasing"):
        normalized_degradation_auc([(0.25, 0.01), (0.25, 0.02)])


def _curve(unit: str, drops: list[float], ratios=DEFAULT_RATIOS) -> list[dict]:
    return [
        {
            "status": "COMPLETED",
            "pruning_unit": unit,
            "architectural_region": "neck",
            "original_channels": 64,
            "requested_pruning_ratio": requested,
            "actual_pruning_ratio": requested,
            "map50_95": 0.5 - drop,
            "accuracy_degradation": drop,
        }
        for requested, drop in zip(ratios, drops)
    ]


def test_rankings_sort_complete_curves_descending_and_leave_incomplete_unranked():
    rows = _curve("sensitive", [0.01, 0.03, 0.06, 0.10])
    rows += _curve("robust", [0.00, 0.01, 0.02, 0.03])
    rows += _curve("missing", [0.01, 0.02, 0.03], ratios=DEFAULT_RATIOS[:3])

    ranking = build_unit_rankings(rows, DEFAULT_RATIOS)

    ranked = [row for row in ranking if row["rank"] is not None]
    assert [(row["pruning_unit"], row["rank"]) for row in ranked] == [
        ("sensitive", 1),
        ("robust", 2),
    ]
    incomplete = next(row for row in ranking if row["pruning_unit"] == "missing")
    assert incomplete["curve_status"] == "INCOMPLETE"
    assert incomplete["rank"] is None
    assert incomplete["normalized_auc_sensitivity"] is None
