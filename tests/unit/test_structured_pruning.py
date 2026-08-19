import pytest
import torch
from torch import nn

from infrared_detection.compression.pruning import (
    compute_channel_importance,
    minimum_weight_scores,
    rank_filters_by_minimum_weight,
    validate_structural_reduction,
)


class _TinyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.head = nn.Conv2d(4, 2, 1)

    def forward(self, x):
        return self.head(self.conv(x))


def test_channel_importance_returns_one_score_per_output_channel():
    model = _TinyNetwork()
    scores = compute_channel_importance(model, criterion="l1")

    assert scores["conv"].shape == (4,)
    assert scores["head"].shape == (2,)


def test_minimum_weight_scores_are_mean_squared_filter_weights():
    conv = nn.Conv2d(1, 3, kernel_size=2, bias=False)
    conv.weight.data.copy_(torch.tensor([
        [[[1.0, 1.0], [1.0, 1.0]]],
        [[[0.0, 0.0], [0.0, 2.0]]],
        [[[2.0, 2.0], [2.0, 2.0]]],
    ]))

    scores = minimum_weight_scores(conv)

    assert scores.tolist() == pytest.approx([1.0, 1.0, 4.0])


def test_minimum_weight_ranking_uses_original_index_as_tie_breaker():
    conv = nn.Conv2d(1, 3, kernel_size=2, bias=False)
    conv.weight.data.copy_(torch.tensor([
        [[[1.0, 1.0], [1.0, 1.0]]],
        [[[0.0, 0.0], [0.0, 2.0]]],
        [[[2.0, 2.0], [2.0, 2.0]]],
    ]))

    assert rank_filters_by_minimum_weight(conv) == (
        (0, pytest.approx(1.0)),
        (1, pytest.approx(1.0)),
        (2, pytest.approx(4.0)),
    )


def test_structural_reduction_requires_fewer_parameters():
    before = {"parameter_count": 100, "serialized_bytes": 400}
    after = {"parameter_count": 60, "serialized_bytes": 240}

    summary = validate_structural_reduction(before, after)

    assert summary["parameter_reduction"] == pytest.approx(0.4)
    assert summary["serialized_reduction"] == pytest.approx(0.4)
