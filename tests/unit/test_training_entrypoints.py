import argparse
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

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


def test_four_baseline_configs_select_distinct_models_and_outputs():
    configs = {
        name: yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
        for name, path in {
            "yolov8n": "configs/models/yolov8n.yaml",
            "yolov8m": "configs/models/yolov8.yaml",
            "resnet34": "configs/models/faster_rcnn_resnet34.yaml",
            "resnet50": "configs/models/faster_rcnn.yaml",
        }.items()
    }

    assert configs["yolov8n"]["model"]["name"] == "yolov8n"
    assert configs["yolov8m"]["model"]["name"] == "yolov8m"
    assert configs["resnet34"]["model"]["backbone"] == "resnet34"
    assert configs["resnet50"]["model"]["backbone"] == "resnet50"

    output_roots = {
        configs[name]["checkpoint"]["resume_from"]
        for name in configs
    }
    assert len(output_roots) == 4


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


def test_yolo_training_dry_run_uses_checkpoint_directory_from_model_config(tmp_path, monkeypatch):
    from infrared_detection.models.yolov8 import training as yolov8_training

    source_config = yaml.safe_load((ROOT / "configs/models/yolov8.yaml").read_text(encoding="utf-8"))
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "camel.yaml").write_text("path: .\n", encoding="utf-8")
    checkpoint_dir = tmp_path / "configured-checkpoints"
    source_config["model"]["device"] = "cpu"
    source_config["data"]["dataset_path"] = str(dataset_dir)
    source_config["checkpoint"]["resume_from"] = str(checkpoint_dir)
    source_config["export"]["onnx"] = False
    config_path = tmp_path / "yolov8.yaml"
    config_path.write_text(yaml.safe_dump(source_config), encoding="utf-8")
    monkeypatch.setattr(yolov8_training, "YOLO", lambda *_: pytest.fail("dry-run must not train"))

    output = yolov8_training.train_yolov8(config_path=config_path, name="baseline", dry_run=True)

    assert output == checkpoint_dir.resolve() / "baseline"


def test_faster_rcnn_training_uses_configured_checkpoint_root():
    from infrared_detection.models.faster_rcnn.training import resolve_checkpoint_root

    config = {"checkpoint": {"resume_from": "artifacts/checkpoints/resnet34"}}

    assert resolve_checkpoint_root(config) == Path("artifacts/checkpoints/resnet34")
