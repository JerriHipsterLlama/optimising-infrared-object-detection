import torch
from torch import nn

from infrared_detection.evaluation.model_stats import collect_model_stats


def test_collect_model_stats_counts_parameters_and_bytes():
    model = nn.Linear(4, 3)

    stats = collect_model_stats(model)

    assert stats["parameter_count"] == 15
    assert stats["parameter_bytes"] == 15 * 4
    assert stats["nonzero_parameter_count"] > 0
