from __future__ import annotations

import pytest
import torch
from torch import nn

from infrared_detection.compression.pruning.importance import (
    l1_filter_scores,
    rank_output_channels,
)
from infrared_detection.compression.pruning.unit_discovery import requested_prune_count


def test_l1_ranking_sums_absolute_filter_weights_and_breaks_ties_by_index():
    conv = nn.Conv2d(1, 4, 1, bias=False)
    with torch.no_grad():
        conv.weight[:, 0, 0, 0] = torch.tensor([-3.0, 1.0, -1.0, 2.0])

    assert l1_filter_scores(conv).tolist() == [3.0, 1.0, 1.0, 2.0]
    assert [index for index, _score in rank_output_channels(conv)] == [1, 2, 3, 0]


@pytest.mark.parametrize(
    ("channels", "ratio", "expected"),
    [
        (16, 0.125, 2),
        (16, 0.25, 4),
        (16, 0.375, 6),
        (16, 0.50, 8),
        (80, 0.125, 10),
    ],
)
def test_requested_prune_count_uses_dense_output_width(channels, ratio, expected):
    assert requested_prune_count(channels, ratio) == expected


@pytest.mark.parametrize(
    ("channels", "ratio", "message"),
    [(1, 0.5, "at least two"), (8, 0.0, "between zero and one"), (8, 1.0, "between zero and one")],
)
def test_requested_prune_count_rejects_invalid_requests(channels, ratio, message):
    with pytest.raises(ValueError, match=message):
        requested_prune_count(channels, ratio)


def test_ranking_rejects_unknown_importance_criterion():
    with pytest.raises(ValueError, match="Supported importance criteria"):
        rank_output_channels(nn.Conv2d(1, 2, 1), criterion="unknown")
