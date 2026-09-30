from pathlib import Path

import torch
from torch import nn
import yaml

from infrared_detection.evaluation.single_layer_performance_screening import (
    ScreeningAdapters,
    run_single_layer_performance_screening,
)


ROOT = Path(__file__).resolve().parents[2]
APPROVED_LAYERS = [
    "model.4.m.1.cv1.conv",
    "model.5.conv",
    "model.6.m.0.cv1.conv",
    "model.6.m.1.cv1.conv",
    "model.7.conv",
    "model.8.cv2.conv",
    "model.8.m.0.cv1.conv",
    "model.9.cv2.conv",
    "model.12.m.0.cv1.conv",
    "model.15.m.0.cv1.conv",
    "model.16.conv",
    "model.18.cv2.conv",
    "model.18.m.0.cv1.conv",
    "model.19.conv",
    "model.21.cv2.conv",
    "model.21.m.0.cv1.conv",
]


def test_yolov8n_filterwise_configs_match_approved_candidates():
    config_paths = (
        ROOT / "configs/experiments/single_layer_performance_yolov8n_rtx.yaml",
        ROOT / "configs/experiments/single_layer_performance_yolov8n_orin.yaml",
    )

    for path in config_paths:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["screening"]["layer_patterns"] == APPROVED_LAYERS


def test_filterwise_curve_keeps_per_filter_rows_for_selected_layers(tmp_path):
    checkpoint = tmp_path / "dense.pt"
    dataset = tmp_path / "dataset.yaml"
    config_path = tmp_path / "screening.yaml"
    checkpoint.write_bytes(b"checkpoint")
    dataset.write_text("path: data\n", encoding="utf-8")
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"checkpoint": str(checkpoint)},
                "data": {"dataset_yaml": str(dataset), "split": "val"},
                "experiment": {
                    "model_variant": "yolov8n",
                    "hardware_label": "unit-test",
                    "image_size": 32,
                    "output_dir": str(tmp_path / "artifacts"),
                },
                "runtime": {"device": "cpu", "precision": "fp32", "conf": 0.25, "iou": 0.6},
                "screening": {"layer_patterns": ["layer_a"]},
            }
        ),
        encoding="utf-8",
    )
    model = nn.Module()
    model.layer_a = nn.Conv2d(1, 4, 1, bias=False)
    with torch.no_grad():
        model.layer_a.weight[:, 0, 0, 0] = torch.tensor([2.0, 4.0, 1.0, 3.0])
    adapters = ScreeningAdapters(
        load_model=lambda path: model,
        evaluate=lambda candidate, config: {
            "map50_95": 0.5,
            "map50": 0.6,
            "precision": 0.7,
            "recall": 0.8,
        },
        prune_filter=lambda candidate, layer, index, config: candidate,
        profile=lambda candidate, config: {"latency_p50_ms": 9.0},
        stats=lambda candidate: {},
        save_checkpoint=lambda candidate, path: Path(path).write_bytes(b"checkpoint"),
    )

    rows = run_single_layer_performance_screening(config_path, adapters=adapters)
    candidates = [row for row in rows if row.get("stage") == "single_layer"]

    assert [row["filter_rank"] for row in candidates] == [1, 2, 3]
    assert [row["filters_remaining"] for row in candidates] == [3, 2, 1]
    assert all(row["layer"] == "layer_a" for row in candidates)
    assert all(row["status"] == "completed" for row in candidates)
    assert all(row["map50_95"] == 0.5 and row["latency_p50_ms"] == 9.0 for row in candidates)
