import torch
from torch import nn

import pytest

from infrared_detection.benchmarking.jetson import (
    JetsonToolsUnavailable,
    benchmark_tensorrt_engine,
    _materialize_trtexec_engine,
)
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


def test_materialize_trtexec_engine_unwraps_ultralytics_metadata(tmp_path):
    engine_path = tmp_path / "wrapped.engine"
    metadata = b'{"description":"test"}'
    raw_engine = b"RAW_TENSORRT_ENGINE"
    engine_path.write_bytes(len(metadata).to_bytes(4, "little") + metadata + raw_engine)

    with _materialize_trtexec_engine(engine_path) as raw_path:
        assert raw_path.read_bytes() == raw_engine

    assert not raw_path.exists()
