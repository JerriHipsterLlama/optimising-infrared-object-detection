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

### Cluster-pruning execution

Run the cluster-size screen separately from model sensitivity and filterwise screening. It compares a baseline with structural probes over the 16 approved Backbone/Neck layers, the configured cluster sizes (including 8), and the configured probe ratio. Each row records validation metrics and parameter-count reduction; this screen does not fine-tune candidates, build engines, or measure Jetson latency.

```powershell
python apps/evaluate_cluster_pruning.py --config configs/experiments/cluster_pruning_rtx_screening.yaml --dry-run
```

Use `--dry-run` to inspect the planned baseline and cluster probes without loading a checkpoint or accessing the dataset. A normal run writes `candidates.csv` and `manifest.json` under `runs/experiments/cluster_pruning_screening/`. Filterwise per-filter accuracy/latency curves remain available through `apps/single_layer_performance_screening.py`; model sensitivity remains available through `apps/yolov8_accuracy_sensitivity.py`.

### RTX compression-matrix screening

The compression matrix creates ONNX artifacts only; target-specific engine building and accuracy/latency testing are separate stages. Dense and structured-pruned candidates start from the dense checkpoint. The matrix writes FP32, FP16, and INT8 ONNX variants for input size 352, the approved candidate layers, cluster size 8, and configured pruning ratios.

Install the optional NVIDIA ModelOpt dependencies in the active environment to enable FP16/INT8 conversion:

```powershell
python -m pip install -e ".[compression]"
```

FP32 conversion simply copies the exported graph. FP16 uses ModelOpt ONNX autocast; INT8 uses ModelOpt ONNX quantization with entropy calibration. The configured calibration images come from `data.calibration_image_dir` (currently the training split), are streamed in deterministic order, letterboxed to the ONNX input shape, converted BGR→RGB, and normalized to [0, 1]. The test split is rejected as a calibration source. Calibration settings are controlled by `quantization.calibration_method` and `quantization.calibration_samples`.

Run a planning check first:

```powershell
$env:PYTHONPATH="$PWD\\src"
python apps/evaluate_compression_matrix.py --config configs/experiments/rtx_compression_matrix.yaml --dry-run
```

Run the local RTX screening with:

```powershell
$env:PYTHONPATH="$PWD\\src"
python apps/evaluate_compression_matrix.py --config configs/experiments/rtx_compression_matrix.yaml
```

The run writes ONNX files, `manifest.json`, and `results.csv` under `runs/experiments/rtx_compression_matrix/`. Each candidate has one shared source ONNX graph and deterministic per-precision output directories. Individual conversion failures are recorded without discarding successful sibling variants. Engine building, validation accuracy/recall, and latency are reported by the separate target-testing workflow, with final deployment claims measured on the Jetson Orin Nano.

TensorRT 11.1 engine building consumes these ONNX graphs; TensorRT runtime precision flags are not a replacement for ONNX FP16/INT8 conversion. FP16 and INT8 ONNX preparation is performed through ModelOpt, independently from TensorRT engine construction.

### Single-layer performance-response screening

The replacement screening workflow independently prunes each selected convolutional layer from its dense checkpoint, one MinimumWeight-ranked filter at a time, until one filter remains. Run the model and hardware combinations with:

```powershell
python apps/single_layer_performance_screening.py --config configs/experiments/single_layer_performance_yolov8n_rtx.yaml
python apps/single_layer_performance_screening.py --config configs/experiments/single_layer_performance_yolov8m_rtx.yaml
python apps/single_layer_performance_screening.py --config configs/experiments/single_layer_performance_yolov8n_orin.yaml
python apps/single_layer_performance_screening.py --config configs/experiments/single_layer_performance_yolov8m_orin.yaml
```

Each configuration writes a resumable `results.csv` beneath `runs/experiments/single_layer_performance_screening/<model>/<hardware>/`. Its leading columns show the candidate, status, filters remaining, mAP50–95, and latency so progress can be inspected while the sweep runs. Latency is direct PyTorch CUDA forward latency on the named device; this sensitivity workflow deliberately produces no ONNX, TensorRT, or other deployment exports.

### YOLOv8n structural-pruning accuracy sensitivity

Run the standalone FP32 workstation screen with:

```powershell
.venv\Scripts\python.exe apps\yolov8_accuracy_sensitivity.py --config configs\experiments\yolov8n_accuracy_sensitivity.yaml
```

The tool evaluates the dense validation baseline, dynamically audits every convolution against the Torch-Pruning dependency graph, and tests independently valid units at ratios `0.125`, `0.25`, `0.375`, and `0.50`. Every `(unit, ratio)` reloads the original checkpoint, so pruning never accumulates. Mechanically dependent BN and downstream input channels may change, but a group that removes another meaningful convolution output is recorded as `GROUPED` and is not evaluated.

The inspected four-class YOLOv8n checkpoint exposes 39 units that remain independent at all four ratios: 27 in the backbone/neck and 12 hidden convolutions in the Detect towers. Runtime discovery remains authoritative. Detect validation preserves the number of scales, prediction ranks and widths, class count, regression representation, and output metadata.

Results are written incrementally to `results.csv`, with full dependency-group JSON, requested and achieved channel counts, validation metrics, accuracy degradation, and point sensitivity. `unit_sensitivity_ranking.csv` contains normalized degradation-AUC rankings for complete curves. Matching terminal rows resume without reevaluation; `ERROR` rows retry from a fresh dense checkpoint. No fine-tuning occurs, and no pruned candidate model is saved.

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
