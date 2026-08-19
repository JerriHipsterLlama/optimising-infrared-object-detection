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

Plan the two-stage candidate workflow on any host with `--dry-run`. A default non-dry run is allowed to claim `jetson_orin_nano` results only when the process verifies that it is running on a Jetson Orin. Mixed RTX screening plus remote-Orin execution must use an injected remote adapter through `run_cluster_evaluation`; the default CLI fails closed on an RTX host instead of labelling local CUDA work as Orin.

```powershell
python apps/evaluate_cluster_pruning.py --config configs/experiments/cluster_pruning_evaluation.yaml --dry-run
```

`targets.orin_target` is logical provenance and must remain `jetson_orin_nano`. `targets.orin_execution_device` is the Ultralytics device argument used for Orin evaluation and fine-tuning and defaults to the valid CUDA selector `'0'`; the logical target label is never passed as a CUDA device. RTX measurements remain probe-screening evidence only. The configured safe layers are concrete `Conv2d` paths, and production pruning selects complete low-importance channel clusters before topology-preserving fine-tuning.

The workflow writes `candidates.csv` and `manifest.json` under the configured `runs/experiments/cluster_pruning_evaluation/` directory. Every export is staged under its candidate directory so Ultralytics cannot overwrite an artifact adjacent to the source checkpoint. Rows record `export_validation_status`; only candidates that pass reloaded-topology parameter reduction and same-format serialized-size reduction can be selected. Functional exported-model inference parity remains a separate validation task. Transfer each surviving global candidate's TensorRT engine to the Orin, build it there if necessary, and measure it with the native benchmark in `deploy/jetson/`. Write each native result back into that same experiment directory, using a distinct filename such as `orin-global-cluster-16-ratio-0.20.json`.

On the Orin, build an engine from a candidate ONNX export when an engine was not transferred:

```bash
trtexec \
  --onnx=runs/experiments/cluster_pruning_evaluation/global-cluster-16-ratio-0.20/candidate.onnx \
  --saveEngine=runs/experiments/cluster_pruning_evaluation/global-cluster-16-ratio-0.20/candidate.engine \
  --fp16 --workspace=1024 --verbose
```

Then run the native batch-1 benchmark against that actual engine:

```bash
./build/jetson/jetson_benchmark \
  --engine runs/experiments/cluster_pruning_evaluation/global-cluster-16-ratio-0.20/candidate.engine \
  --input-dir data/camel/images/test --warmup 20 --iterations 100 \
  --output-json runs/experiments/cluster_pruning_evaluation/orin-global-cluster-16-ratio-0.20.json
```

The native JSON must be annotated with the candidate identifier before it is merged. For example, on the Orin:

```bash
python3 -c "import json; p='runs/experiments/cluster_pruning_evaluation/orin-global-cluster-16-ratio-0.20.json'; data=json.load(open(p)); data['candidate_id']='global-cluster-16-ratio-0.2'; open(p, 'w').write(json.dumps(data, indent=2) + '\\n')"
```

Add the tegrastats handoff fields to the same JSON before copying it to the host. The merge accepts this explicit schema; `device` or `target` must be exactly `jetson_orin_nano`:

```json
{
  "candidate_id": "global-cluster-16-ratio-0.2",
  "device": "jetson_orin_nano",
  "target": "jetson_orin_nano",
  "latency_p50_ms": 8.1,
  "latency_p95_ms": 8.6,
  "fps": 118.7,
  "peak_memory_mb": 742.0,
  "power_w": 8.4,
  "energy_mj_per_inference": 70.8,
  "temperature_c": 52.0
}
```

Capture the supplemental values during the native run, keeping the raw handoff beside the JSON:

```bash
tegrastats --interval 1000 --logfile runs/experiments/cluster_pruning_evaluation/tegrastats-global-cluster-16-ratio-0.20.log
```

After transferring that JSON to the experiment directory, merge it on the host. The merge requires the exact candidate ID, exact `jetson_orin_nano` provenance, and positive finite numeric `latency_p50_ms` and `latency_p95_ms` measurements before setting `hardware_benchmarked=True`. It updates only native hardware fields (`latency_*`, FPS, memory, power, energy, and temperature); validation mAP and serialized-size fields remain unchanged. The benchmark path and provenance are bound into the row and manifest before both `candidates.csv` and `manifest.json` are rewritten.

```powershell
python -c "import json; from pathlib import Path; from infrared_detection.evaluation.cluster_workflow import merge_jetson_metrics; output=Path('runs/experiments/cluster_pruning_evaluation'); merge_jetson_metrics(json.loads((output / 'manifest.json').read_text(encoding='utf-8'))['rows'], output / 'orin-global-cluster-16-ratio-0.20.json')"
```

Merge native results for every final global candidate before reading `selected_candidate_ids` from the manifest. The manifest remains unselected until a valid Orin-provenance merge marks a structurally/size-validated feasible global row as hardware-benchmarked. Only measured Orin results rank final candidates; RTX results are screening evidence only.

### RTX compression-matrix screening

The local RTX workflow builds and evaluates dense and structured-pruned TensorRT variants. The pruning variants always start from the dense checkpoint; filterwise results are analysed manually to choose the cluster size and candidate layers before the matrix is run. The current matrix tests FP32 and FP16 at input size 352, the recommended candidate layers, cluster size 8, and the configured pruning ratios. INT8 is deferred while the TensorRT 11.1 ModelOpt workflow is developed.

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

The run writes `manifest.json` and `results.csv` under `runs/experiments/rtx_compression_matrix/`. It records one row per precision and ratio, continues after individual failures, and selects the highest completed structured-pruning FP32 ratio within `pruning.allowed_map50_95_drop` of the dense FP32 baseline. RTX results are preliminary screening evidence; final deployment claims must be measured on the Jetson Orin Nano.

The compression-matrix `runtime.evaluation_device` is intentionally `cpu` for TensorRT engine validation on Windows. Ultralytics' TensorRT backend uses this selector to initialize the `.engine` on CUDA while avoiding intermittent PyTorch numeric-device visibility failures; latency is still measured with the raw TensorRT engine through `trtexec`.

TensorRT 11.1 no longer accepts the legacy `trtexec --fp16`, `--int8`, and `--calib` flags. FP16 is prepared through the ModelOpt autocast boundary before engine building. INT8 is intentionally excluded from the current configuration and will require a separate ModelOpt calibration-data workflow.

### Single-layer performance-response screening

The replacement screening workflow independently prunes each selected convolutional layer from its dense checkpoint, one MinimumWeight-ranked filter at a time, until one filter remains. Run the model and hardware combinations with:

```powershell
python apps/single_layer_performance_screening.py --config configs/experiments/single_layer_performance_yolov8n_rtx.yaml
python apps/single_layer_performance_screening.py --config configs/experiments/single_layer_performance_yolov8m_rtx.yaml
python apps/single_layer_performance_screening.py --config configs/experiments/single_layer_performance_yolov8n_orin.yaml
python apps/single_layer_performance_screening.py --config configs/experiments/single_layer_performance_yolov8m_orin.yaml
```

Each configuration writes a resumable `results.csv` beneath `runs/experiments/single_layer_performance_screening/<model>/<hardware>/`. Its leading columns show the candidate, status, filters remaining, mAP50–95, and latency so progress can be inspected while the sweep runs. Latency is direct PyTorch CUDA forward latency on the named device; this sensitivity workflow deliberately produces no ONNX, TensorRT, or other deployment exports.

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
