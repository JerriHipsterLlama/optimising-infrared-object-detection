from pathlib import Path

import yaml


def test_jetson_benchmark_config_declares_reproducible_measurement_fields():
    path = Path("configs/deployment/jetson_orin_nano.yaml")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["platform"] == "jetson_orin_nano"
    assert config["batch_size"] == 1
    assert config["warmup"] > 0
    assert config["iterations"] >= 100
    assert {"preprocess_mean_ms", "inference_mean_ms", "postprocess_mean_ms", "latency_p50_ms", "latency_p95_ms", "fps"}.issubset(config["metrics"])
