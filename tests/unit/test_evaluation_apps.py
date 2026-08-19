import subprocess
import sys
import importlib.util
from pathlib import Path

import yaml


def test_evaluation_apps_expose_help():
    for app in ("evaluate.py", "export.py", "benchmark.py", "report.py", "evaluate_cluster_pruning.py"):
        result = subprocess.run(
            [sys.executable, f"apps/{app}", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_cluster_pruning_app_exposes_filterwise_mode():
    result = subprocess.run(
        [sys.executable, "apps/evaluate_cluster_pruning.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--filterwise" in result.stdout
    assert "--full-curve" in result.stdout


def test_matrix_app_exposes_config_and_dry_run():
    result = subprocess.run(
        [sys.executable, "apps/evaluate_compression_matrix.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    assert "--dry-run" in result.stdout


def test_single_layer_performance_app_exposes_config_only():
    result = subprocess.run(
        [sys.executable, "apps/single_layer_performance_screening.py", "--help"],
        capture_output=True, text=True, check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    assert "--full-curve" not in result.stdout


def test_single_layer_performance_app_passes_its_progress_function_as_on_result(monkeypatch, tmp_path):
    app_path = Path(__file__).resolve().parents[2] / "apps" / "single_layer_performance_screening.py"
    spec = importlib.util.spec_from_file_location("single_layer_performance_screening_app", app_path)
    assert spec is not None and spec.loader is not None
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    config_path = tmp_path / "screening.yaml"
    config_path.write_text("runtime:\n  device: '0'\n", encoding="utf-8")
    received: dict = {}

    class FakeAdapters:
        @staticmethod
        def defaults(config):
            return "adapters"

    def fake_run(config_path, **kwargs):
        received["config_path"] = config_path
        received.update(kwargs)

    monkeypatch.setattr(app, "ScreeningAdapters", FakeAdapters)
    monkeypatch.setattr(app, "run_single_layer_performance_screening", fake_run)
    monkeypatch.setattr(sys, "argv", [str(app_path), "--config", str(config_path)])

    app.main()

    assert received["config_path"] == str(config_path)
    assert received["adapters"] == "adapters"
    assert received["on_result"] is app._progress
    assert "progress_callback" not in received


def test_single_layer_performance_configs_have_approved_values_and_isolated_outputs():
    root = Path(__file__).resolve().parents[2]
    expected_patterns = [
        "model.6.m.*.cv1.conv", "model.6.m.*.cv2.conv", "model.8.m.*.cv1.conv", "model.8.m.*.cv2.conv",
        "model.12.m.*.cv1.conv", "model.12.m.*.cv2.conv", "model.18.m.*.cv1.conv", "model.18.m.*.cv2.conv",
        "model.21.m.*.cv1.conv", "model.21.m.*.cv2.conv",
    ]
    expected_configs = {
        "single_layer_performance_yolov8n_rtx.yaml": (
            "yolov8n", "rtx3070", "models/checkpoints/yolov8/train3/weights/best.pt",
            "runs/experiments/single_layer_performance_screening/yolov8n/rtx3070",
        ),
        "single_layer_performance_yolov8m_rtx.yaml": (
            "yolov8m", "rtx3070", "models/checkpoints/yolov8/train4/weights/best.pt",
            "runs/experiments/single_layer_performance_screening/yolov8m/rtx3070",
        ),
        "single_layer_performance_yolov8n_orin.yaml": (
            "yolov8n", "jetson_orin_nano", "models/checkpoints/yolov8/train3/weights/best.pt",
            "runs/experiments/single_layer_performance_screening/yolov8n/jetson_orin_nano",
        ),
        "single_layer_performance_yolov8m_orin.yaml": (
            "yolov8m", "jetson_orin_nano", "models/checkpoints/yolov8/train4/weights/best.pt",
            "runs/experiments/single_layer_performance_screening/yolov8m/jetson_orin_nano",
        ),
    }

    output_dirs = set()
    for filename, (variant, hardware, checkpoint, output_dir) in expected_configs.items():
        config = yaml.safe_load((root / "configs" / "experiments" / filename).read_text(encoding="utf-8"))

        assert config["model"]["checkpoint"] == checkpoint
        assert config["data"] == {"dataset_yaml": "data/camel/camel.yaml", "split": "val"}
        assert config["experiment"]["model_variant"] == variant
        assert config["experiment"]["hardware_label"] == hardware
        assert config["experiment"]["image_size"] == 352
        assert config["experiment"]["batch_size"] == 1
        assert config["experiment"]["output_dir"] == output_dir
        assert config["runtime"] == {"device": "0", "precision": "fp16", "conf": 0.25, "iou": 0.6}
        assert config["screening"]["warmup"] == 20
        assert config["screening"]["iterations"] == 100
        assert config["screening"]["layer_patterns"] == expected_patterns
        output_dirs.add(config["experiment"]["output_dir"])

    assert len(output_dirs) == 4
