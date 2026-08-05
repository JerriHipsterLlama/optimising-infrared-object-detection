"""Generate the self-contained Kaggle filterwise-pruning notebook."""

from __future__ import annotations

import argparse
import json
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
        "source": [source],
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
            code_cell("# Configuration is embedded in this notebook."),
            markdown_cell("## Input validation"),
            code_cell("# Validate checkpoint, dataset YAML, images, labels, and Kaggle GPU."),
            markdown_cell("## Pruning engine"),
            code_cell("# Perform dependency-aware minimum-L1 filter pruning."),
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
