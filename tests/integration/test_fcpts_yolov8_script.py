"""Tests for YOLO FCPTS calibration CLI script."""

import subprocess
import sys
from pathlib import Path


def test_fcpts_yolov8_script_help_runs():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "src" / "infrared_detection" / "compression" / "pruning" / "fcpts" / "yolo.py"

    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )

    assert result.returncode == 0
    assert "Run FCPTS calibration on a YOLOv8 checkpoint" in result.stdout


def test_fcpts_yolov8_script_uses_config_in_dry_run(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "src" / "infrared_detection" / "compression" / "pruning" / "fcpts" / "yolo.py"
    config_path = tmp_path / "fcpts_test.yaml"
    config_path.write_text(
        "training:\n"
        "  epochs: 3\n"
        "  lr: 0.0002\n"
        "runtime:\n"
        "  device: cpu\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--checkpoint",
            "models\\baseline\\yolov8\\train\\weights\\best.pt",
            "--config",
            str(config_path),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )

    assert result.returncode == 0
    assert "epochs=3" in result.stdout
    assert "lr=0.0002" in result.stdout
    assert "device=cpu" in result.stdout
