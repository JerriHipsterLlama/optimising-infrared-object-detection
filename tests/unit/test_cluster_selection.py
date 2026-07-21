import torch

from infrared_detection.compression.pruning.cluster_selection import (
    ClusterSpec,
    plan_low_importance_clusters,
)


def test_plan_uses_the_lowest_scoring_complete_cluster():
    scores = {"model.2.conv": torch.tensor([0.9, 0.1, 0.2, 0.8, 0.3])}

    assert plan_low_importance_clusters(scores, 2, set()) == [
        ClusterSpec("model.2.conv", 2, (1, 2))
    ]


def test_plan_skips_protected_layers_and_incomplete_tail():
    scores = {
        "model.2.conv": torch.tensor([0.1, 0.2, 0.9, 1.0, 2.0]),
        "model.22.cv3": torch.tensor([0.1, 0.2, 0.3, 0.4]),
    }

    assert plan_low_importance_clusters(scores, 2, {"model.22.cv3"}) == [
        ClusterSpec("model.2.conv", 2, (0, 1))
    ]
