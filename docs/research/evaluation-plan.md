# Research Direction and Evaluation Plan

## Working title

**A controlled evaluation of low-precision quantization, structured pruning, and knowledge distillation for memory-constrained infrared object detection on the NVIDIA Jetson Orin Nano**

## Research problem

Modern object detectors can be accurate but difficult to deploy on memory-limited edge hardware. Compression methods are often evaluated independently, with different models, datasets, training procedures, and hardware. This makes it difficult to determine which technique actually provides the best deployment trade-off.

This project will compare low-precision quantization (LPQ), structured channel pruning, and knowledge distillation under a common experimental protocol for infrared object detection. The study will use the same detector families, dataset, input resolution, accuracy evaluation, and Jetson deployment environment wherever possible.

The project is primarily an empirical and deployment-oriented study. It does not need to claim a new pruning algorithm. Its contribution is a controlled, reproducible comparison that separates the effects of compression mechanisms from the effects of accuracy-recovery training.

## Central research question

> Under matched accuracy and deployment constraints, which compression strategy, or combination of strategies, provides the best accuracy-memory-latency trade-off for infrared object detection on the NVIDIA Jetson Orin Nano?

## Sub-questions

1. How much accuracy is lost when LPQ, structured pruning, or both are applied without knowledge distillation?
2. How much of the lost accuracy can KD recover for each compressed model?
3. Which method provides the best reduction in model size, peak memory, latency, and energy per inference?
4. Do the conclusions remain consistent between YOLO and Faster R-CNN, or are they architecture-dependent?
5. Does the method that looks best using parameter count or FLOPs also perform best in TensorRT on the Orin Nano?

## Experimental design

### Models and data

- Primary dataset: CAMEL infrared object-detection dataset.
- Primary detector: YOLOv8n, because it is already implemented in this project and is suitable for edge deployment.
- Secondary detector: Faster R-CNN with MobileNetV3-FPN, once the YOLO pipeline is stable.
- Teacher: the full-precision trained detector.
- Student: the compressed detector being evaluated.
- All variants must use the same train, validation, and test splits.
- All variants must use the same image resolution, batch size during inference, confidence threshold, IoU threshold, and post-processing configuration.

### Factorial comparison

The experiment treats compression and KD as separate factors. This avoids presenting KD as if it were directly equivalent to pruning or quantization: pruning and quantization reduce the deployed model, while KD primarily helps train a smaller or lower-precision model.

| Variant | Compression | KD |
|---|---|---|
| A | Dense FP32 reference | No |
| B | LPQ/INT8 | No |
| C | Structured channel pruning | No |
| D | Structured channel pruning + LPQ/INT8 | No |
| E | LPQ/INT8 | Yes |
| F | Structured channel pruning | Yes |
| G | Structured channel pruning + LPQ/INT8 | Yes |

FP16 TensorRT deployment should also be reported as the practical dense deployment baseline. INT8 should preferably use quantization-aware training when the accuracy drop from post-training quantization is too large.

### Structured pruning protocol

The pruning branch must physically rebuild the model rather than merely zeroing weights. The protocol will:

1. identify dependency-connected convolution channels;
2. protect or conservatively prune sensitive layers, including the stem and detection head during early experiments;
3. prune progressively in small steps rather than removing the final target in one operation;
4. fine-tune after every pruning step;
5. use KD from the dense teacher during recovery experiments;
6. preserve hardware-friendly channel dimensions, preferably multiples of 8 or 16;
7. record both the intended structural reduction and the resulting TensorRT engine measurements.

Importance criteria should be compared or selected explicitly, rather than silently assuming that weight magnitude is sufficient. Candidate criteria include BN scaling factors, filter norms, Taylor importance, and activation-weight sensitivity.

## Metrics for success

Success is multi-dimensional. No method will be considered successful merely because it removes parameters or lowers FLOPs.

### 1. Detection accuracy

Primary metrics:

- mAP50;
- mAP50-95;
- precision;
- recall;
- per-class AP;
- AP for small, medium, and large objects where the dataset supports the required object-size analysis.

The main accuracy constraint should be mAP50-95 relative to the dense FP32 baseline.

**Minimum acceptable accuracy outcome:** no more than a 2 percentage-point absolute drop in mAP50-95 for a model claimed to be deployment-successful. Results with a larger drop may still be reported as a compression trade-off, but should not be presented as accuracy-preserving.

### 2. Model size and storage

Report:

- parameter count;
- trainable parameter count;
- serialized checkpoint size;
- ONNX file size;
- TensorRT engine size;
- FP32, FP16, and INT8 storage where applicable.

**Minimum meaningful size outcome:** at least 25% reduction in serialized deployment artifact size compared with the dense FP32 model, while satisfying the accuracy constraint.

### 3. Memory usage

Measure on the Jetson Orin Nano:

- peak system RAM during TensorRT engine building;
- peak system RAM during inference;
- peak GPU/unified memory during inference;
- model-weight memory;
- activation/workspace memory where available from TensorRT profiling.

**Minimum meaningful memory outcome:** at least 20% reduction in peak inference memory compared with the dense FP16 deployment baseline, with no accuracy drop greater than the defined 2-point limit.

### 4. Inference performance

Measure at batch size 1 using a warmed-up TensorRT engine:

- end-to-end latency;
- preprocessing latency;
- model execution latency;
- post-processing/NMS latency;
- p50 latency;
- p95 latency;
- FPS;
- TensorRT engine build time.

At least 100 timed inferences should be collected after warm-up. The reported value should be the median over repeated measurement runs, with interquartile range or standard deviation.

**Minimum meaningful performance outcome:** at least 20% reduction in median model latency, or at least 1.25x throughput, compared with the dense FP16 TensorRT baseline, while satisfying the accuracy constraint.

### 5. Energy and thermal behaviour

Where measurement tools permit, report:

- average power during inference;
- peak power;
- energy per inference;
- temperature after a sustained workload;
- performance stability over a longer run.

**Preferred outcome:** lower energy per inference without a disproportionate increase in temperature or throttling. Energy per inference is more informative than instantaneous power alone:

`energy per inference = average power / inferences per second`

### 6. Pareto efficiency

The final comparison should identify Pareto-optimal models rather than selecting a single winner using an arbitrary weighted score. A model is Pareto-dominated if another model has equal or better accuracy, lower memory, and lower latency.

The primary plots should show:

- mAP50-95 versus latency;
- mAP50-95 versus peak memory;
- mAP50-95 versus engine size;
- energy per inference versus mAP50-95.

## Proposed success definition

The project will be considered successful if it produces:

1. a reproducible baseline and deployment measurement protocol;
2. at least one compressed model with no more than a 2-point absolute mAP50-95 drop;
3. at least 25% smaller serialized deployment artifact;
4. at least 20% lower peak inference memory or at least 20% lower median latency;
5. a statistically supported comparison showing when KD improves each compression method;
6. an explanation of any disagreement between theoretical metrics such as FLOPs and measured Jetson performance.

If no method meets all thresholds, that is still a valid result provided the study explains the accuracy-efficiency frontier and identifies the limiting factor. The thresholds are acceptance criteria for a practically useful compressed model, not assumptions about what the experiments must achieve.

## Reproducibility and fairness controls

- Use fixed random seeds for training and pruning experiments.
- Use identical train/validation/test splits.
- Keep the detector input resolution and inference batch size fixed.
- Use the same TensorRT, CUDA, JetPack, and operating-system versions for all Jetson measurements.
- Report whether measurements include preprocessing and post-processing.
- Warm up the device before timing.
- Run multiple timing repetitions and report dispersion.
- Do not compare PyTorch GPU timing against TensorRT timing as if they were equivalent.
- Report both theoretical metrics and real hardware metrics.
- Keep the dense teacher out of the deployment memory and latency measurements.

## Literature positioning

Prior work already combines subsets of these techniques, particularly pruning with KD or quantization with KD. The intended gap is narrower and more defensible: a controlled comparison of the techniques and combinations under a common infrared-detection and Jetson Orin Nano deployment protocol.

The study should therefore claim an empirical hardware-aware evaluation, not a wholly new compression algorithm, unless the experiments later reveal a genuinely novel method.

## Recommended implementation order

1. Establish the dense YOLOv8 FP32, FP16, and INT8 baselines.
2. Build an automated CAMEL evaluation and metric-reporting pipeline.
3. Implement and validate structured channel pruning on YOLOv8.
4. Add progressive fine-tuning without KD.
5. Add KD as an independent training factor.
6. Add LPQ/INT8 to the dense and pruned variants.
7. Export all variants to TensorRT.
8. Benchmark the variants on the Jetson Orin Nano.
9. Repeat the most informative comparisons with Faster R-CNN.

