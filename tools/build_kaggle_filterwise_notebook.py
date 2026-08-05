"""Generate the self-contained Kaggle filterwise-pruning notebook."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any


def markdown_cell(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}


def code_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [textwrap.dedent(source).strip() + "\n"],
    }


def build_notebook(output_path: Path) -> None:
    """Write a valid, self-contained notebook skeleton for Kaggle."""

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "cells": [
            markdown_cell("# Kaggle filterwise pruning sensitivity\n\nSelf-contained YOLOv8n layer sensitivity experiment."),
            markdown_cell("## Setup"),
            code_cell("!pip install -q ultralytics==8.4.7 torch-pruning pandas matplotlib pyyaml"),
            markdown_cell("## Configuration"),
            code_cell(
                '''
                from __future__ import annotations

                import csv
                import hashlib
                import json
                import platform
                import random
                from datetime import datetime, timezone
                from pathlib import Path

                import matplotlib.pyplot as plt
                import numpy as np
                import pandas as pd
                import torch
                import yaml
                from ultralytics import YOLO

                CONFIG = {
                    "input_root": "/kaggle/input/camel-filterwise-input",
                    "checkpoint_relpath": "best.pt",
                    "dataset_yaml_relpath": "dataset.yaml",
                    "output_dir": "/kaggle/working/filterwise_sensitivity",
                    "imgsz": 352,
                    "split": "val",
                    "device": 0,
                    "conf": 0.25,
                    "iou": 0.6,
                    "seed": 7,
                    "max_map50_95_drop": 0.02,
                    "max_recall_drop": 0.02,
                    "full_curve": True,
                    "retry_failed": False,
                    "resume_dir": None,
                    "protected_prefixes": ["model.22"],
                    "milestone_removals": [1, 8, 16, 32],
                }

                def set_seed(seed):
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)

                set_seed(CONFIG["seed"])
                '''
            ),
            markdown_cell("## Input validation"),
            code_cell(
                '''
                def file_identity(path):
                    path = Path(path).resolve()
                    digest = hashlib.sha256()
                    with path.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(chunk)
                    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


                def validate_inputs(config):
                    root = Path(config["input_root"]).resolve()
                    if any(".npy" in str(value).lower() for value in config.values() if isinstance(value, str)):
                        raise ValueError("Duplicate .npy image files are not supported by this notebook input package.")
                    checkpoint_path = root / config["checkpoint_relpath"]
                    dataset_yaml = root / config["dataset_yaml_relpath"]
                    if not checkpoint_path.is_file():
                        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
                    if not dataset_yaml.is_file():
                        raise FileNotFoundError(f"Missing dataset YAML: {dataset_yaml}")
                    dataset = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8")) or {}
                    val_path = dataset.get("val")
                    if not isinstance(val_path, str) or not val_path:
                        raise ValueError("dataset.yaml must define a non-empty 'val' path for Ultralytics validation.")
                    image_dir = Path(val_path)
                    if not image_dir.is_absolute():
                        image_dir = (dataset_yaml.parent / image_dir).resolve()
                    label_dir = image_dir.parent.parent / "labels" / image_dir.name
                    if not image_dir.is_dir() or not any(image_dir.glob("*.*")):
                        raise FileNotFoundError(f"Validation images are missing: {image_dir}")
                    if not label_dir.is_dir():
                        raise FileNotFoundError(f"Validation labels are missing: {label_dir}")
                    if not torch.cuda.is_available():
                        raise RuntimeError("A Kaggle GPU is required. Enable GPU acceleration in Notebook settings.")
                    return {
                        "root": root,
                        "checkpoint_path": checkpoint_path.resolve(),
                        "dataset_yaml": dataset_yaml.resolve(),
                        "image_dir": image_dir,
                        "label_dir": label_dir,
                        "gpu_name": torch.cuda.get_device_name(config["device"]),
                        "torch_version": torch.__version__,
                    }


                def _json_safe(value):
                    if isinstance(value, Path):
                        return str(value)
                    if isinstance(value, (np.integer, np.floating)):
                        return value.item()
                    raise TypeError(f"Cannot serialize {type(value).__name__}")


                def write_artifacts(state):
                    output_dir = Path(state["config"]["output_dir"])
                    output_dir.mkdir(parents=True, exist_ok=True)
                    rows = state["rows"]
                    fields = sorted({field for row in rows for field in row})
                    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(handle, fieldnames=fields)
                        writer.writeheader()
                        writer.writerows(rows)
                    manifest = {key: value for key, value in state.items() if key != "rows"}
                    manifest["rows"] = rows
                    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_safe) + "\\n", encoding="utf-8")
                    progress = {"completed": sorted(state["completed"]), "failed": sorted(state["failed"]), "updated_at": datetime.now(timezone.utc).isoformat()}
                    (output_dir / "progress.json").write_text(json.dumps(progress, indent=2) + "\\n", encoding="utf-8")


                def evaluate_dense_baseline(config):
                    validated = validate_inputs(config)
                    wrapper = YOLO(str(validated["checkpoint_path"]))
                    metrics = wrapper.val(
                        data=str(validated["dataset_yaml"]), split=config["split"], imgsz=config["imgsz"],
                        device=config["device"], conf=config["conf"], iou=config["iou"], verbose=False,
                    )
                    return {
                        "candidate_id": "baseline",
                        "stage": "baseline",
                        "status": "completed",
                        "checkpoint_path": str(validated["checkpoint_path"]),
                        "map50": float(metrics.box.map50),
                        "map50_95": float(metrics.box.map),
                        "precision": float(metrics.box.mp),
                        "recall": float(metrics.box.mr),
                        "parameter_count": int(sum(parameter.numel() for parameter in wrapper.model.parameters())),
                    }
                '''
            ),
            markdown_cell("## Pruning engine"),
            code_cell(
                '''
                import torch_pruning as tp


                def get_module(model, layer_name):
                    module = dict(model.named_modules()).get(layer_name)
                    if not isinstance(module, torch.nn.Conv2d):
                        raise ValueError(f"{layer_name!r} is not a Conv2d module")
                    return module


                def example_input_for(model, imgsz, device):
                    first_conv = next(module for module in model.modules() if isinstance(module, torch.nn.Conv2d))
                    cuda_device = f"cuda:{device}" if isinstance(device, int) else device
                    return torch.zeros((1, first_conv.in_channels, imgsz, imgsz), device=cuda_device)


                def dependency_graph(model, config):
                    return tp.DependencyGraph().build_dependency(
                        model,
                        example_inputs=example_input_for(model, config["imgsz"], config["device"]),
                    )


                def discover_prunable_layers(model):
                    excluded = tuple(CONFIG["protected_prefixes"])
                    candidates = []
                    for name, module in model.named_modules():
                        if not isinstance(module, torch.nn.Conv2d) or module.out_channels <= 1:
                            continue
                        if name.startswith(excluded):
                            continue
                        try:
                            graph = dependency_graph(model, CONFIG)
                            group = graph.get_pruning_group(module, tp.prune_conv_out_channels, idxs=[0])
                            if group.check_pruning_group():
                                candidates.append(name)
                        except Exception as exc:
                            print(f"Skipping unsupported layer {name}: {exc}")
                    return candidates


                def minimum_l1_filter(module):
                    scores = module.weight.detach().abs().sum(dim=(1, 2, 3))
                    return int(scores.argmin().item())


                def prune_output_channel_with_dependencies(model, layer_name, index, config):
                    module = get_module(model, layer_name)
                    graph = dependency_graph(model, config)
                    group = graph.get_pruning_group(module, tp.prune_conv_out_channels, idxs=[index])
                    if not group.check_pruning_group():
                        raise RuntimeError(f"Dependency pruning is unsafe for {layer_name} filter {index}")
                    group.prune()
                    return model


                def load_dense_wrapper(config):
                    validated = validate_inputs(config)
                    wrapper = YOLO(str(validated["checkpoint_path"]))
                    cuda_device = f"cuda:{config['device']}" if isinstance(config["device"], int) else config["device"]
                    wrapper.model.to(cuda_device).eval()
                    return wrapper, validated


                def evaluate_model(wrapper, config):
                    validated = validate_inputs(config)
                    metrics = wrapper.val(
                        data=str(validated["dataset_yaml"]), split=config["split"], imgsz=config["imgsz"],
                        device=config["device"], conf=config["conf"], iou=config["iou"], verbose=False,
                    )
                    return {
                        "map50": float(metrics.box.map50),
                        "map50_95": float(metrics.box.map),
                        "precision": float(metrics.box.mp),
                        "recall": float(metrics.box.mr),
                    }


                def run_layer_sweep(layer_name, config, state):
                    wrapper, validated = load_dense_wrapper(config)
                    model = wrapper.model
                    filters_before = get_module(model, layer_name).out_channels
                    filters_after = get_module(model, layer_name).out_channels
                    while filters_after > 1:
                        module = get_module(model, layer_name)
                        filters_before_step = module.out_channels
                        filters_removed = filters_before - filters_before_step + 1
                        candidate_key = f"{layer_name}:{filters_removed}"
                        if candidate_key in state["completed"]:
                            model = prune_output_channel_with_dependencies(model, layer_name, minimum_l1_filter(module), config)
                            wrapper.model = model
                            filters_after = get_module(model, layer_name).out_channels
                            continue
                        try:
                            index = minimum_l1_filter(module)
                            model = prune_output_channel_with_dependencies(model, layer_name, index, config)
                            wrapper.model = model
                            metrics = evaluate_model(wrapper, config)
                            filters_after = get_module(model, layer_name).out_channels
                            row = {
                                "candidate_id": f"filterwise-{layer_name.replace('.', '-')}-filters-{filters_removed}",
                                "stage": "filterwise",
                                "status": "completed",
                                "layer": layer_name,
                                "filters_before": filters_before,
                                "filters_removed": filters_removed,
                                "filters_after": filters_after,
                                "pruned_filter_index": index,
                                "checkpoint_path": str(validated["checkpoint_path"]),
                                **metrics,
                            }
                            state["rows"].append(row)
                            state["completed"].add(candidate_key)
                            write_artifacts(state)
                        except Exception as exc:
                            state["rows"].append({
                                "candidate_id": f"filterwise-{layer_name.replace('.', '-')}-filters-{filters_removed}",
                                "stage": "filterwise", "status": "failed", "layer": layer_name,
                                "filters_before": filters_before, "filters_removed": filters_removed,
                                "error": f"{type(exc).__name__}: {exc}",
                            })
                            state["failed"].add(candidate_key)
                            write_artifacts(state)
                            break
                    return state
                '''
            ),
            markdown_cell("## Run or resume"),
            code_cell("# Persist results.csv, manifest.json, and progress.json after every result."),
            markdown_cell("## Analysis"),
            code_cell("# Write plots and layer-summary artifacts."),
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("notebooks/kaggle_filterwise_sensitivity.ipynb"))
    args = parser.parse_args()
    build_notebook(args.output)


if __name__ == "__main__":
    main()
