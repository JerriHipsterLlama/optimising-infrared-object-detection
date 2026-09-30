import argparse
import subprocess
import sys
from pathlib import Path

from apps.train import build_parser


ROOT = Path(__file__).resolve().parents[2]


def test_training_cli_keeps_yolov8_and_faster_rcnn_families():
    parser = build_parser()
    subparsers = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))

    assert set(subparsers.choices) == {"yolov8", "faster-rcnn"}
    assert subparsers.choices["yolov8"].get_default("handler").__name__ == "_train_yolov8"
    assert subparsers.choices["faster-rcnn"].get_default("handler").__name__ == "_train_faster_rcnn"


def test_training_default_configs_exist():
    parser = build_parser()
    subparsers = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))

    assert subparsers.choices["yolov8"].get_default("config") == "configs/models/yolov8.yaml"
    assert subparsers.choices["faster-rcnn"].get_default("config") == "configs/models/faster_rcnn.yaml"
    assert (ROOT / "configs/models/yolov8.yaml").is_file()
    assert (ROOT / "configs/models/faster_rcnn.yaml").is_file()


def test_sensitivity_entrypoint_and_config_remain_present():
    from infrared_detection.compression.pruning import (
        build_yolo_dependency_graph,
        discover_pruning_units,
        serialize_dependency_group,
    )
    from infrared_detection.evaluation.accuracy_sensitivity import (
        load_sensitivity_config,
        run_accuracy_sensitivity,
    )

    app = ROOT / "apps/yolov8_accuracy_sensitivity.py"
    config = ROOT / "configs/experiments/yolov8n_accuracy_sensitivity.yaml"
    assert app.is_file()
    assert load_sensitivity_config(config).split == "val"
    assert callable(run_accuracy_sensitivity)
    assert all(callable(api) for api in (build_yolo_dependency_graph, discover_pruning_units, serialize_dependency_group))


def test_training_and_sensitivity_help_do_not_start_workflows():
    training = subprocess.run(
        [sys.executable, str(ROOT / "apps/train.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    sensitivity = subprocess.run(
        [sys.executable, str(ROOT / "apps/yolov8_accuracy_sensitivity.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    assert "yolov8" in training and "faster-rcnn" in training
    assert "--config" in sensitivity and "Sensitivity experiment YAML" in sensitivity
