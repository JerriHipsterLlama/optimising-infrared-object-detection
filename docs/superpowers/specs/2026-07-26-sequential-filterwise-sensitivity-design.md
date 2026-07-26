# Sequential Filter-wise Sensitivity Evaluation

## Purpose

Reduce the runtime of the filter-wise YOLO pruning experiment while preserving the complete one-filter-at-a-time research protocol for the final selected layers.

## Scope

The workflow will operate on these five representative layers in the filter-wise experiment configuration:

- `model.0.conv` — early, high-resolution feature extraction;
- `model.2.cv2.conv` — C2f block at the 128-channel stage;
- `model.4.cv2.conv` — C2f block at the 256-channel stage;
- `model.6.cv2.conv` — C2f block at the 512-channel stage;
- `model.8.cv2.conv` — deep C2f block at the final feature scale.

For each layer it will start from the dense baseline and repeatedly:

1. compute Minimum Weight scores for the currently remaining filters;
2. select the lowest-scoring filter;
3. physically prune that filter and dependent channels;
4. save the resulting candidate checkpoint;
5. record model statistics and evaluate accuracy.

The next iteration will recompute Minimum Weight scores on the newly pruned model. This produces a sequential curve rather than independently selecting all filters from the original baseline.

## Evaluation modes

Rapid screening will support early stopping when accuracy is effectively zero, using a configurable threshold and optional consecutive-failure count. The research mode will disable early stopping and continue until one filter remains, producing the complete curve.

Every pruning step will record filter counts, accuracy, parameter count, checkpoint path, and pruning status. TensorRT export and latency profiling will be limited to configured alignment points and nearby off-alignment control points during screening, rather than every filter count.

## Data flow

```text
dense baseline
  -> MW rank
  -> prune one filter
  -> save checkpoint
  -> accuracy/statistics
  -> optional TensorRT latency profile
  -> repeat until stop condition
```

The accuracy sweep remains exhaustive in research mode. The latency sweep is deliberately sparse during screening because TensorRT export and engine benchmarking dominate runtime; final Jetson measurements will be performed on selected layers and cluster-size candidates.

## Stop and profiling policy

- Screening mode stops when `map50_95` falls below a configurable near-zero threshold for the configured number of consecutive iterations.
- Research mode ignores that threshold and continues to one remaining filter.
- Latency candidates include expected hardware-aligned counts, such as removal counts around multiples of 8, plus neighbouring off-alignment controls.
- RTX measurements are preliminary; the Jetson Orin Nano remains authoritative for the final hardware-aware cluster size.

## Outputs

The existing `candidates.csv` and `manifest.json` formats will be extended with sequential iteration metadata, early-stop reasons, and whether a candidate was selected for hardware profiling. Existing cluster-pruning evaluation remains unchanged.

## Acceptance criteria

- Selected layers produce candidates from the baseline down to one remaining filter when research mode is enabled.
- Minimum Weight scores are recomputed after every physical filter removal.
- Rapid screening can stop without generating later candidates after the configured accuracy-collapse condition.
- Accuracy and structural statistics are recorded for every generated candidate.
- Latency profiling is not required for every candidate in rapid screening.
- Existing focused cluster-pruning tests continue to pass.
