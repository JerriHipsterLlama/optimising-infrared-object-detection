import subprocess
import sys
from pathlib import Path

import pytest

from infrared_detection.models.yolov8 import training


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_count_dataset_images_supports_jpg_and_npy_without_double_counting(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "frame_001.jpg").write_bytes(b"jpg")
    (image_dir / "frame_001.npy").write_bytes(b"npy")
    (image_dir / "frame_002.jpg").write_bytes(b"jpg")

    assert training.count_dataset_images(image_dir) == 2


def test_resolve_dataset_yaml_is_repo_relative(tmp_path):
    yaml_path = tmp_path / "data" / "camel.yaml"
    yaml_path.parent.mkdir()
    yaml_path.write_text("path: .\n", encoding="utf-8")

    assert training.resolve_dataset_yaml("data/camel.yaml", tmp_path) == yaml_path.resolve()


def test_requested_cuda_device_fails_actionably_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(training.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA device 0 requested"):
        training.resolve_training_device(0)


def test_cpu_device_remains_available_without_cuda(monkeypatch):
    monkeypatch.setattr(training.torch.cuda, "is_available", lambda: False)

    assert training.resolve_training_device("cpu") == "cpu"


def test_train_app_help_lists_model_subcommands():
    result = subprocess.run(
        [sys.executable, "apps/train.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "yolov8" in result.stdout
    assert "faster-rcnn" in result.stdout
