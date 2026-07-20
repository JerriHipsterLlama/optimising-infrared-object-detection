# Infrared Object Detection Research Pipeline

This repository evaluates quantization, structured pruning, and knowledge distillation for infrared object detection on the NVIDIA Jetson Orin Nano.

## Setup

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

Local datasets belong in `data/`. Generated checkpoints, ONNX files, TensorRT engines, and prediction artifacts belong in `artifacts/` and are gitignored.

## Dataset preparation

Dataset configuration is in `configs/dataset/camel.yaml`. Dataset conversion and validation utilities are in `scripts/dataset/`:

```powershell
python scripts/dataset/organize_dataset.py
python scripts/dataset/convert_labels_to_pascal_format.py --split all
python scripts/dataset/check_empty_labels.py
```

## Training

```powershell
python apps/train.py yolov8 --config configs/models/yolov8.yaml
python apps/train.py faster-rcnn --config configs/models/faster_rcnn.yaml
```

Training outputs should use `artifacts/checkpoints/` or an explicitly configured output directory.

## Compression

```powershell
python apps/compress.py matrix --config configs/experiments/compression_matrix.yaml --dry-run
```

The compression library is under `src/infrared_detection/compression/`, with pruning, quantization, and distillation kept as separate factors.

## Evaluation and reporting

```powershell
python apps/evaluate.py --input artifacts/predictions/raw_metrics.json --output artifacts/predictions/metrics.json
python apps/report.py --input reports/experiment_rows.json --output reports/pareto.json
```

The evaluation plan is [docs/research/evaluation-plan.md](docs/research/evaluation-plan.md). Core metrics include mAP50, mAP50-95, precision, recall, serialized artifact size, memory, latency, FPS, energy, and Pareto efficiency.

## Export and Jetson benchmarking

Export a model through the narrow adapter in `src/infrared_detection/export/`:

```powershell
python apps/export.py --model artifacts/checkpoints/best.pt --format onnx --imgsz 320
```

The Python-side TensorRT adapter is available through `apps/benchmark.py`. The native Jetson benchmark is documented under `deploy/jetson/` and is the authoritative path for batch-1 latency, p50/p95 timing, memory, power, and thermal measurements.

The native benchmark consumes real test images and can preserve raw TensorRT output tensors for offline Python post-processing. Use the native output for deployment-performance metrics and the Python evaluation pipeline for mAP, precision, recall, and class-level analysis.

## Repository layout

```text
src/infrared_detection/  reusable Python package
apps/                    thin research workflow entry points
configs/                 dataset, model, compression, and deployment config
scripts/                 dataset, development, and reporting utilities
deploy/jetson/           native TensorRT/CUDA benchmark
tests/                   unit, integration, and regression tests
data/                    local datasets, ignored by Git
artifacts/               generated model and prediction artifacts, ignored by Git
reports/                 selected experiment summaries
```
