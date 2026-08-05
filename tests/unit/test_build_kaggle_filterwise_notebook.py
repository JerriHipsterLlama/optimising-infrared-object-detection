from __future__ import annotations

import json

from tools.build_kaggle_filterwise_notebook import build_notebook


def test_build_notebook_creates_self_contained_kaggle_notebook(tmp_path):
    output = tmp_path / "kaggle_filterwise_sensitivity.ipynb"

    build_notebook(output)

    notebook = json.loads(output.read_text(encoding="utf-8"))
    source = "\n".join(line for cell in notebook["cells"] for line in cell.get("source", []))
    headings = [
        "Setup",
        "Configuration",
        "Input validation",
        "Pruning engine",
        "Run or resume",
        "Analysis",
    ]

    assert notebook["nbformat"] == 4
    assert "Kaggle filterwise pruning sensitivity" in source
    assert "git clone" not in source
    assert "torch-pruning" in source
    assert "results.csv" in source
    assert "progress.json" in source
    assert all(heading in source for heading in headings)


def test_notebook_embeds_validation_and_durable_artifact_writers(tmp_path):
    output = tmp_path / "notebook.ipynb"

    build_notebook(output)

    source = "\n".join(line for cell in json.loads(output.read_text(encoding="utf-8"))["cells"] for line in cell.get("source", []))

    assert "def validate_inputs(config):" in source
    assert "def write_artifacts(state):" in source
    assert "def evaluate_dense_baseline(config):" in source
    assert "checkpoint_path" in source
    assert ".npy" in source


def test_notebook_embeds_complete_filterwise_sweep_contract(tmp_path):
    output = tmp_path / "notebook.ipynb"

    build_notebook(output)

    source = "\n".join(line for cell in json.loads(output.read_text(encoding="utf-8"))["cells"] for line in cell.get("source", []))

    assert "def discover_prunable_layers(model):" in source
    assert "def minimum_l1_filter(module):" in source
    assert "def run_layer_sweep(layer_name, config, state):" in source
    assert "DependencyGraph" in source
    assert "filters_after > 1" in source
    assert "filters_removed" in source


def test_notebook_embeds_resume_and_layer_summary_contract(tmp_path):
    output = tmp_path / "notebook.ipynb"

    build_notebook(output)

    source = "\n".join(line for cell in json.loads(output.read_text(encoding="utf-8"))["cells"] for line in cell.get("source", []))

    assert "def load_resume_state(config):" in source
    assert "def plot_layer_sensitivity(rows, output_dir):" in source
    assert "def build_layer_summary(rows, config):" in source
    assert "layer_summary.csv" in source
    assert "max_recall_drop" in source
