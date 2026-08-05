# Kaggle Filterwise Sensitivity Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one self-contained Kaggle Jupyter notebook that produces complete, resumable 1-filter-at-a-time sensitivity curves for every supported YOLOv8n convolution layer.

**Architecture:** The notebook contains all executable logic: input validation, layer discovery, dependency-aware structural pruning, Ultralytics validation, durable result writes, resume checks, plots, and ranked recommendations. The only external inputs are the small Kaggle dataset package containing validation images/labels/YAML and `best.pt`; no source-code clone or duplicate NPY files are required.

**Tech Stack:** Python 3, Jupyter/Colab-compatible `.ipynb`, PyTorch, Ultralytics 8.4.7, Torch-Pruning, NumPy, Pandas, Matplotlib, PyYAML.

## Global Constraints

- Create the notebook at `notebooks/kaggle_filterwise_sensitivity.ipynb`.
- Embed all experiment code and configuration in the notebook; do not import this repository at runtime.
- Exclude duplicate NPY image files from the documented Kaggle input package.
- Default to a complete curve: remove exactly one output filter per evaluation until one filter remains.
- Keep each layer independent by reloading the dense checkpoint before its sweep.
- Use minimum-L1 output-filter importance and dependency-aware structural pruning.
- Do not prune the YOLO `Detect` head or unsupported graph-dependent modules.
- Rewrite `results.csv`, `manifest.json`, and `progress.json` after every result row.
- Treat Kaggle GPU latency only as sensitivity evidence; do not make Jetson latency or energy claims.
- Preserve the existing uncommitted `configs/experiments/cluster_pruning_rtx_screening.yaml` edit.

---

### Task 1: Add a testable notebook generator and configuration contract

**Files:**
- Create: `tools/build_kaggle_filterwise_notebook.py`
- Create: `tests/unit/test_build_kaggle_filterwise_notebook.py`
- Create: `notebooks/kaggle_filterwise_sensitivity.ipynb`

**Interfaces:**
- Produces: `build_notebook(output_path: Path) -> None`
- Produces: a valid notebook with ordered sections `Setup`, `Configuration`, `Input validation`, `Pruning engine`, `Run or resume`, and `Analysis`.
- Consumes: no repository modules at notebook runtime.

- [ ] **Step 1: Write the failing notebook-structure test**

```python
import json

from tools.build_kaggle_filterwise_notebook import build_notebook


def test_build_notebook_creates_self_contained_kaggle_notebook(tmp_path):
    output = tmp_path / "kaggle_filterwise_sensitivity.ipynb"

    build_notebook(output)

    notebook = json.loads(output.read_text(encoding="utf-8"))
    source = "\n".join(
        line for cell in notebook["cells"] for line in cell.get("source", [])
    )
    assert notebook["nbformat"] == 4
    assert "Kaggle filterwise pruning sensitivity" in source
    assert "git clone" not in source
    assert "torch-pruning" in source
    assert "results.csv" in source
    assert "progress.json" in source
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\test_build_kaggle_filterwise_notebook.py -q`

Expected: FAIL because `tools.build_kaggle_filterwise_notebook` does not exist.

- [ ] **Step 3: Implement the minimal notebook builder**

```python
def build_notebook(output_path: Path) -> None:
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}},
        "cells": [
            markdown_cell("# Kaggle filterwise pruning sensitivity"),
            code_cell("!pip install -q ultralytics==8.4.7 torch-pruning pandas matplotlib pyyaml"),
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
```

Include cells for every required section, all configuration values, and all implementation functions as literal notebook code. Do not create an external runtime module for the notebook to import.

- [ ] **Step 4: Run the unit test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\test_build_kaggle_filterwise_notebook.py -q`

Expected: PASS.

- [ ] **Step 5: Generate the tracked notebook and inspect its JSON**

Run: `.venv\Scripts\python.exe tools\build_kaggle_filterwise_notebook.py --output notebooks\kaggle_filterwise_sensitivity.ipynb`

Run: `.venv\Scripts\python.exe -m json.tool notebooks\kaggle_filterwise_sensitivity.ipynb > $null`

Expected: The notebook is valid JSON and contains all required section headings.

- [ ] **Step 6: Commit**

```powershell
git add tools/build_kaggle_filterwise_notebook.py tests/unit/test_build_kaggle_filterwise_notebook.py notebooks/kaggle_filterwise_sensitivity.ipynb
git commit -m "feat: add self-contained Kaggle sensitivity notebook"
```

### Task 2: Implement input validation, dense baseline, and durable artifacts in the notebook

**Files:**
- Modify: `tools/build_kaggle_filterwise_notebook.py`
- Modify: `notebooks/kaggle_filterwise_sensitivity.ipynb`
- Modify: `tests/unit/test_build_kaggle_filterwise_notebook.py`

**Interfaces:**
- Produces notebook functions: `validate_inputs(config) -> dict`, `write_artifacts(state) -> None`, `evaluate_dense_baseline(config) -> dict`.
- Consumes: Kaggle input directory with `best.pt`, `dataset.yaml`, `images/val`, and `labels/val`.
- Produces: durable `results.csv`, `manifest.json`, and `progress.json` under `OUTPUT_DIR`.

- [ ] **Step 1: Extend the failing test for required input/artifact code**

```python
def test_notebook_embeds_validation_and_durable_artifact_writers(tmp_path):
    output = tmp_path / "notebook.ipynb"
    build_notebook(output)
    source = "\n".join(
        line for cell in json.loads(output.read_text())["cells"] for line in cell.get("source", [])
    )
    assert "def validate_inputs(config):" in source
    assert "def write_artifacts(state):" in source
    assert "baseline" in source
    assert "checkpoint_path" in source
    assert ".npy" in source
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\test_build_kaggle_filterwise_notebook.py -q`

Expected: FAIL because the required notebook functions are not embedded yet.

- [ ] **Step 3: Add notebook configuration and artifact functions**

Embed a configuration cell with these values:

```python
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
    "resume_dir": None,
}
```

Implement `validate_inputs` so it rejects missing checkpoint/YAML, absent validation image/label directories, absent CUDA, and any configured NPY path. Implement `write_artifacts` using `csv.DictWriter` and `json.dumps(..., indent=2)` so every completed row is immediately recoverable. Record package paths, GPU name, library versions, configuration, and dense baseline metrics in the manifest.

- [ ] **Step 4: Run the focused test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\test_build_kaggle_filterwise_notebook.py -q`

Expected: PASS.

- [ ] **Step 5: Regenerate and validate notebook JSON**

Run: `.venv\Scripts\python.exe tools\build_kaggle_filterwise_notebook.py --output notebooks\kaggle_filterwise_sensitivity.ipynb`

Run: `.venv\Scripts\python.exe -m json.tool notebooks\kaggle_filterwise_sensitivity.ipynb > $null`

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add tools/build_kaggle_filterwise_notebook.py tests/unit/test_build_kaggle_filterwise_notebook.py notebooks/kaggle_filterwise_sensitivity.ipynb
git commit -m "feat: add resumable Kaggle sensitivity artifacts"
```

### Task 3: Add independent, dependency-aware 1-filter pruning sweeps

**Files:**
- Modify: `tools/build_kaggle_filterwise_notebook.py`
- Modify: `notebooks/kaggle_filterwise_sensitivity.ipynb`
- Modify: `tests/unit/test_build_kaggle_filterwise_notebook.py`

**Interfaces:**
- Produces notebook functions: `discover_prunable_layers(model) -> list[str]`, `minimum_l1_filter(module) -> int`, `run_layer_sweep(layer_name, config, state) -> None`.
- Consumes: a dense YOLO checkpoint and the configuration/artifact functions from Task 2.
- Produces: one independent result row per `layer_name` and `filters_removed` value from `1` to `filters_before - 1`.

- [ ] **Step 1: Extend the failing test for one-filter structural sweep code**

```python
def test_notebook_embeds_complete_filterwise_sweep_contract(tmp_path):
    output = tmp_path / "notebook.ipynb"
    build_notebook(output)
    source = "\n".join(
        line for cell in json.loads(output.read_text())["cells"] for line in cell.get("source", [])
    )
    assert "def discover_prunable_layers(model):" in source
    assert "def minimum_l1_filter(module):" in source
    assert "def run_layer_sweep(layer_name, config, state):" in source
    assert "DependencyGraph" in source
    assert "filters_after > 1" in source
    assert "filters_removed" in source
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\test_build_kaggle_filterwise_notebook.py -q`

Expected: FAIL because pruning discovery and sweep functions are not embedded yet.

- [ ] **Step 3: Add the pruning engine cells**

Implement the following behaviour inside the notebook:

```python
def minimum_l1_filter(module):
    scores = module.weight.detach().abs().sum(dim=(1, 2, 3))
    return int(scores.argmin().item())

def run_layer_sweep(layer_name, config, state):
    model = load_dense_model(config)
    while get_module(model, layer_name).out_channels > 1:
        module = get_module(model, layer_name)
        index = minimum_l1_filter(module)
        model = prune_output_channel_with_dependencies(model, layer_name, index, config["imgsz"])
        metrics = evaluate_model(model, config)
        append_result_and_write_artifacts(state, layer_name, module.out_channels - 1, metrics)
```

`discover_prunable_layers` must inspect named modules, keep only supported `Conv2d` modules, exclude the `Detect` head and protected layer names, and dry-run dependency construction before returning a layer. Each `run_layer_sweep` must reload the dense model once at the start of its layer, then prune sequentially so the minimum-L1 filter is recalculated after each removal. Catch a single layer failure, write its failure row, and continue with the next layer.

- [ ] **Step 4: Run the focused test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\test_build_kaggle_filterwise_notebook.py -q`

Expected: PASS.

- [ ] **Step 5: Regenerate the notebook and inspect the pruning source**

Run: `.venv\Scripts\python.exe tools\build_kaggle_filterwise_notebook.py --output notebooks\kaggle_filterwise_sensitivity.ipynb`

Run: `.venv\Scripts\python.exe -c "import json; n=json.load(open('notebooks/kaggle_filterwise_sensitivity.ipynb')); s=''.join(''.join(c.get('source', [])) for c in n['cells']); assert 'DependencyGraph' in s and 'minimum_l1_filter' in s"`

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add tools/build_kaggle_filterwise_notebook.py tests/unit/test_build_kaggle_filterwise_notebook.py notebooks/kaggle_filterwise_sensitivity.ipynb
git commit -m "feat: add complete Kaggle filterwise pruning sweep"
```

### Task 4: Add resume validation, sensitivity plots, and layer recommendations

**Files:**
- Modify: `tools/build_kaggle_filterwise_notebook.py`
- Modify: `notebooks/kaggle_filterwise_sensitivity.ipynb`
- Modify: `tests/unit/test_build_kaggle_filterwise_notebook.py`

**Interfaces:**
- Produces notebook functions: `load_resume_state(config) -> dict`, `plot_layer_sensitivity(rows, output_dir) -> None`, `build_layer_summary(rows, config) -> pandas.DataFrame`.
- Consumes: artifacts produced by Tasks 2 and 3.
- Produces: `plots/<layer>_sensitivity.png` and `layer_summary.csv`.

- [ ] **Step 1: Extend the failing test for resume and analysis cells**

```python
def test_notebook_embeds_resume_and_layer_summary_contract(tmp_path):
    output = tmp_path / "notebook.ipynb"
    build_notebook(output)
    source = "\n".join(
        line for cell in json.loads(output.read_text())["cells"] for line in cell.get("source", [])
    )
    assert "def load_resume_state(config):" in source
    assert "def plot_layer_sensitivity(rows, output_dir):" in source
    assert "def build_layer_summary(rows, config):" in source
    assert "layer_summary.csv" in source
    assert "max_recall_drop" in source
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\test_build_kaggle_filterwise_notebook.py -q`

Expected: FAIL because resume and analysis functions are not embedded yet.

- [ ] **Step 3: Add resume and analysis cells**

Implement `load_resume_state` so it compares checkpoint identity, dataset YAML identity, image size, split, pruning criterion, and layer list before accepting prior result rows. Skip a completed `(layer, filters_removed)` pair; rerun failed rows only when `CONFIG["retry_failed"]` is true.

Implement `plot_layer_sensitivity` to create a two-panel PNG per layer: mAP50-95 and recall versus filters removed, with dense baseline and configured drop thresholds marked. Implement `build_layer_summary` to return one row per layer containing `filters_before`, `max_filters_removed_within_map_limit`, `max_filters_removed_within_recall_limit`, `recommended_max_filters_removed`, and `status`. Set `recommended_max_filters_removed` to the smaller of the mAP and recall limits.

- [ ] **Step 4: Run the focused test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\test_build_kaggle_filterwise_notebook.py -q`

Expected: PASS.

- [ ] **Step 5: Regenerate notebook and run the complete test suite**

Run: `.venv\Scripts\python.exe tools\build_kaggle_filterwise_notebook.py --output notebooks\kaggle_filterwise_sensitivity.ipynb`

Run: `$env:PYTHONPATH=(Get-Location).Path+'\src'; .venv\Scripts\python.exe -m pytest -q`

Expected: PASS with no failures.

- [ ] **Step 6: Commit**

```powershell
git add tools/build_kaggle_filterwise_notebook.py tests/unit/test_build_kaggle_filterwise_notebook.py notebooks/kaggle_filterwise_sensitivity.ipynb
git commit -m "feat: summarize resumable Kaggle sensitivity results"
```

### Task 5: Verify the generated notebook is ready for Kaggle handoff

**Files:**
- Modify: `README.md`
- Modify: `notebooks/kaggle_filterwise_sensitivity.ipynb`

**Interfaces:**
- Produces: a short README subsection explaining the four input items and how to recover Kaggle working-directory outputs.
- Consumes: the completed notebook from Tasks 1-4.

- [ ] **Step 1: Write the failing documentation assertion**

```python
def test_readme_links_to_kaggle_filterwise_notebook():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "kaggle_filterwise_sensitivity.ipynb" in text
    assert "Duplicate NPY" in text
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\test_build_kaggle_filterwise_notebook.py::test_readme_links_to_kaggle_filterwise_notebook -q`

Expected: FAIL because the README does not describe the Kaggle handoff.

- [ ] **Step 3: Add concise Kaggle handoff documentation**

Add a README section named `Kaggle filterwise sensitivity sweep` with:

```markdown
- Upload validation JPG/PNG images, YOLO labels, `dataset.yaml`, and `best.pt` as one private Kaggle Dataset.
- Do not upload duplicate NPY image files.
- Open `notebooks/kaggle_filterwise_sensitivity.ipynb`, update `CONFIG["input_root"]`, and enable a Kaggle GPU.
- Download or version the `filterwise_sensitivity` working-directory output after each session to resume safely.
```

- [ ] **Step 4: Run focused and complete verification**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\test_build_kaggle_filterwise_notebook.py -q`

Run: `$env:PYTHONPATH=(Get-Location).Path+'\src'; .venv\Scripts\python.exe -m pytest -q`

Expected: PASS with no failures.

- [ ] **Step 5: Commit**

```powershell
git add README.md notebooks/kaggle_filterwise_sensitivity.ipynb tests/unit/test_build_kaggle_filterwise_notebook.py
git commit -m "docs: describe Kaggle filterwise notebook handoff"
```
