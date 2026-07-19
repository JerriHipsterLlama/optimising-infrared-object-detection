import pytest

from infrared_detection.common.experiment_config import (
    load_experiment_config,
    resolve_repo_path,
    validate_experiment_config,
)


def test_resolve_repo_path_anchors_relative_paths_to_repository(tmp_path):
    resolved = resolve_repo_path("data/camel/camel.yaml", tmp_path)
    assert resolved == tmp_path / "data" / "camel" / "camel.yaml"


def test_validate_experiment_config_requires_reproducibility_fields():
    with pytest.raises(ValueError, match="dataset_yaml"):
        validate_experiment_config({"model": {"checkpoint": "best.pt"}})


def test_load_experiment_config_preserves_device_and_precision(tmp_path):
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "model:\n"
        "  checkpoint: models/best.pt\n"
        "data:\n"
        "  dataset_yaml: data/camel/camel.yaml\n"
        "runtime:\n"
        "  device: 0\n"
        "  precision: int8\n"
        "experiment:\n"
        "  image_size: 320\n"
        "  num_classes: 4\n"
        "  seed: 7\n"
        "  batch_size: 1\n"
        "  output_dir: runs/experiment\n",
        encoding="utf-8",
    )

    loaded = load_experiment_config(config_path)

    assert loaded["runtime"]["device"] == 0
    assert loaded["runtime"]["precision"] == "int8"
    assert loaded["experiment"]["seed"] == 7
