import json
import subprocess
import sys
import textwrap
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


def _dispatch_train_app_with_stubbed_trainer(
    model_family: str,
    module_name: str,
    trainer_name: str,
    cli_args: list[str],
) -> dict:
    script = textwrap.dedent(
        f"""
        import json
        import runpy
        import sys
        import types
        from pathlib import Path

        repo_root = Path.cwd()
        src_path = str((repo_root / "src").resolve())
        sys.path[:] = [path for path in sys.path if Path(path or ".").resolve() != Path(src_path)]

        trainer_module = types.ModuleType({module_name!r})

        def trainer(**kwargs):
            print(json.dumps(kwargs, sort_keys=True))

        setattr(trainer_module, {trainer_name!r}, trainer)
        sys.modules[{module_name!r}] = trainer_module

        app = runpy.run_path(str(repo_root / "apps" / "train.py"), run_name="train_app")
        assert src_path in sys.path, "apps/train.py must bootstrap the repository src directory"
        args = app["build_parser"]().parse_args({[model_family, *cli_args]!r})
        args.handler(args)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_train_app_yolov8_dispatches_all_supported_overrides_without_install():
    forwarded = _dispatch_train_app_with_stubbed_trainer(
        "yolov8",
        "infrared_detection.models.yolov8.training",
        "train_yolov8",
        [
            "--config", "configs/custom_yolo.yaml",
            "--epochs", "12",
            "--batch-size", "4",
            "--img-size", "640",
            "--resume",
            "--device", "cpu",
            "--seed", "99",
            "--data", "data/custom.yaml",
            "--project", "artifacts/checkpoints/custom",
            "--name", "trial",
            "--dry-run",
        ],
    )

    assert forwarded == {
        "batch_size": 4,
        "config_path": "configs/custom_yolo.yaml",
        "data": "data/custom.yaml",
        "device": "cpu",
        "dry_run": True,
        "epochs": 12,
        "img_size": 640,
        "name": "trial",
        "project": "artifacts/checkpoints/custom",
        "resume": True,
        "seed": 99,
    }


def test_train_app_faster_rcnn_dispatches_all_supported_overrides_without_install():
    forwarded = _dispatch_train_app_with_stubbed_trainer(
        "faster-rcnn",
        "infrared_detection.models.faster_rcnn.training",
        "train_faster_rcnn",
        [
            "--config", "configs/custom_faster_rcnn.yaml",
            "--epochs", "8",
            "--batch-size", "2",
            "--resume",
            "--checkpoint", "artifacts/checkpoints/faster_rcnn/run/weights/last.pt",
            "--name", "resume-run",
            "--no-augment",
        ],
    )

    assert forwarded == {
        "batch_size": 2,
        "checkpoint_path": "artifacts/checkpoints/faster_rcnn/run/weights/last.pt",
        "config_path": "configs/custom_faster_rcnn.yaml",
        "epochs": 8,
        "name": "resume-run",
        "no_augment": True,
        "resume": True,
    }
