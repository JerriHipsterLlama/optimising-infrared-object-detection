import torch
from torch import nn

import pytest

from infrared_detection.benchmarking.jetson import JetsonToolsUnavailable, benchmark_tensorrt_engine
from infrared_detection.evaluation.model_stats import collect_model_stats


def test_collect_model_stats_counts_parameters_and_bytes():
    model = nn.Linear(4, 3)

    stats = collect_model_stats(model)

    assert stats["parameter_count"] == 15
    assert stats["parameter_bytes"] == 15 * 4
    assert stats["nonzero_parameter_count"] > 0


def test_benchmark_reports_clear_diagnostic_when_trtexec_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("infrared_detection.benchmarking.jetson.shutil.which", lambda name: None)
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"engine")

    with pytest.raises(JetsonToolsUnavailable, match="trtexec"):
        benchmark_tensorrt_engine(engine_path, iterations=2, warmup=1)
