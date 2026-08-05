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
