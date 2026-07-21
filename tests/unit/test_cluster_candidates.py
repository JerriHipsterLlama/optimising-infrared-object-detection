import pytest

from infrared_detection.evaluation.cluster_candidates import (
    classify_candidate,
    select_cluster_candidates,
)


@pytest.mark.parametrize(
    ("candidate", "status"),
    [
        (0.4900, "primary_feasible"),
        (0.4899, "exploratory_feasible"),
        (0.4799, "rejected_accuracy"),
    ],
)
def test_classification_uses_absolute_map50_95_limits(candidate, status):
    assert classify_candidate(0.50, candidate) == status


def test_primary_winner_is_smallest_then_fastest_on_orin():
    rows = [
        {
            "status": "primary_feasible",
            "serialized_bytes": 100,
            "latency_p50_ms": 8.0,
            "map50_95": 0.495,
        },
        {
            "status": "primary_feasible",
            "serialized_bytes": 100,
            "latency_p50_ms": 7.0,
            "map50_95": 0.492,
        },
    ]

    assert select_cluster_candidates(rows)["primary"] is rows[1]


def test_missing_latency_sorts_after_measured_latency():
    measured = {
        "status": "primary_feasible",
        "serialized_bytes": 100,
        "latency_p50_ms": 8.0,
        "map50_95": 0.49,
    }
    missing = {
        "status": "primary_feasible",
        "serialized_bytes": 100,
        "map50_95": 0.50,
    }

    assert select_cluster_candidates([missing, measured])["primary"] is measured


def test_exploratory_winner_is_selected_separately():
    exploratory = {
        "status": "exploratory_feasible",
        "serialized_bytes": 120,
        "latency_p50_ms": 10.0,
        "map50_95": 0.48,
    }

    assert select_cluster_candidates([exploratory]) == {
        "primary": None,
        "exploratory": exploratory,
    }
