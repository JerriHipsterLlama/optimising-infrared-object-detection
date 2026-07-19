"""Configuration loading and repository-relative path handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_PATHS = ("model.checkpoint", "data.dataset_yaml")
REQUIRED_EXPERIMENT_FIELDS = (
    "image_size",
    "num_classes",
    "seed",
    "batch_size",
    "output_dir",
)


def _get_nested(config: dict[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def resolve_repo_path(path: str | Path, repo_root: Path) -> Path:
    """Resolve a path without allowing relative paths to depend on cwd."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else (repo_root / candidate).resolve()


def validate_experiment_config(config: dict[str, Any]) -> None:
    """Validate the common fields needed for comparable experiments."""

    if not isinstance(config, dict):
        raise ValueError("Experiment config must be a mapping.")

    for key in REQUIRED_PATHS:
        value = _get_nested(config, key)
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError(f"Missing required config field: {key}")

    experiment = config.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("Missing required config section: experiment")
    for key in REQUIRED_EXPERIMENT_FIELDS:
        if key not in experiment:
            raise ValueError(f"Missing required config field: experiment.{key}")

    runtime = config.get("runtime")
    if not isinstance(runtime, dict) or "device" not in runtime or "precision" not in runtime:
        raise ValueError("Runtime config must define device and precision.")


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML experiment configuration."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    validate_experiment_config(config)
    config["_config_path"] = str(config_path.resolve())
    return config
