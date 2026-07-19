# Research-Pipeline Repository Structure

## Status

Approved design for implementation.

## Objective

Reorganize the repository around the infrared object-detection research workflow: dataset preparation, model training, compression, export, Jetson deployment, benchmarking, and reporting. The reorganization may change imports and command paths. Existing generated artifacts and local datasets must remain outside version control.

## Target architecture

```text
infrared-object-detection/
├── README.md
├── pyproject.toml
├── requirements/
│   ├── training.txt
│   ├── jetson.txt
│   └── development.txt
├── configs/
│   ├── dataset/
│   ├── models/
│   ├── compression/
│   ├── deployment/
│   └── experiments/
├── src/
│   └── infrared_detection/
│       ├── data/
│       ├── models/
│       │   ├── yolov8/
│       │   └── faster_rcnn/
│       ├── training/
│       ├── compression/
│       │   ├── pruning/
│       │   ├── quantization/
│       │   └── distillation/
│       ├── export/
│       ├── evaluation/
│       ├── benchmarking/
│       └── common/
├── apps/
│   ├── train.py
│   ├── evaluate.py
│   ├── compress.py
│   ├── export.py
│   ├── benchmark.py
│   └── report.py
├── deploy/
│   └── jetson/
│       ├── CMakeLists.txt
│       ├── include/
│       ├── src/
│       └── README.md
├── experiments/
│   ├── definitions/
│   ├── manifests/
│   └── schemas/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/
├── docs/
│   ├── research/
│   ├── workflows/
│   ├── deployment/
│   └── decisions/
├── scripts/
│   ├── dataset/
│   ├── development/
│   └── reporting/
├── data/                  # local and gitignored
├── artifacts/             # local and gitignored
│   ├── checkpoints/
│   ├── onnx/
│   ├── tensorrt/
│   └── predictions/
└── reports/               # selected, reproducible outputs
```

## Boundaries

- `src/infrared_detection/` contains reusable library code only.
- `apps/` contains thin command-line entry points and no substantial model logic.
- `configs/` contains declarative configuration, grouped by concern.
- `experiments/` defines the compression matrix and records experiment metadata, not model implementation.
- `deploy/jetson/` contains the native TensorRT/CUDA benchmark runtime and is independently buildable with CMake.
- `scripts/` contains small operational utilities such as dataset conversion and report generation.
- `data/` and `artifacts/` are runtime outputs and are never imported as source packages.
- `tests/` is grouped by test intent rather than by the source folder being tested.

## Data flow

```text
configs + experiment definition
        ↓
apps/train.py
        ↓
artifacts/checkpoints/
        ↓
apps/compress.py ──→ fine-tuning / distillation
        ↓
apps/export.py
        ↓
artifacts/onnx/ and artifacts/tensorrt/
        ↓
deploy/jetson benchmark executable
        ↓
artifacts/predictions/ and benchmark JSON
        ↓
apps/evaluate.py and apps/report.py
        ↓
reports/
```

Accuracy evaluation remains Python-based. Jetson deployment measurements use the native C++ TensorRT runtime and emit machine-readable results that Python can aggregate with detection metrics and Pareto plots.

## Migration rules

1. Consolidate `src/python/` into the single installable package `src/infrared_detection/`.
2. Remove the duplicate top-level `python/` package unless a compatibility shim is required temporarily during migration.
3. Move model-family-specific code under `models/yolov8/` and `models/faster_rcnn/`.
4. Move pruning, LPQ/quantization, and distillation under `compression/`.
5. Move export and TensorRT Python helpers out of generic evaluation modules.
6. Keep dataset conversion utilities separate from reusable dataset loaders.
7. Move generated model files and run outputs to `artifacts/`; update `.gitignore` accordingly.
8. Preserve the existing research evaluation plan and migrate it under `docs/research/` without changing its substance.
9. Update tests and imports as part of each move; do not leave two supported package roots.
10. Add a small smoke-test command that imports the package and validates the experiment configuration before model execution.

## Verification requirements

- All existing tests continue to pass after imports and paths are updated.
- The package can be installed in editable mode from the repository root.
- Each app exposes a documented `--help` command.
- Dataset, training, compression, export, and evaluation modules remain importable independently.
- The Jetson benchmark has a documented build and execution path.
- No tracked source file depends on the old `src.python` or top-level `python` import path.
- Generated artifacts are excluded from version control.

## Out of scope

- Redesigning model architectures.
- Changing the compression algorithms or research protocol.
- Replacing the existing training framework.
- Building a full experiment-tracking service.
- Requiring the Jetson toolchain on the development workstation.
