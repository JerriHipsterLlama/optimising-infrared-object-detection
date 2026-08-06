# RTX all-layer filterwise sensitivity sweep

## Goal

Measure pruning sensitivity for every dependency-safe convolutional layer in the YOLOv8 checkpoint on the local RTX 3070. Each layer is pruned independently and sequentially by one minimum-weight output filter per step. A layer stops once its measured mAP50-95 is below 0.30.

## Configuration

`configs/experiments/filterwise_rtx_screening.yaml` will use automatic layer selection rather than a fixed list and fixed output widths. It will retain `model.22` as protected and set the early-stop metric threshold to `0.30` mAP50-95. The runner invocation must not request `full_curve`, because that mode intentionally suppresses early stopping.

## Runner behavior

The filterwise runner will load the dense baseline first, query its dependency-safe layers, obtain each layer's current output width, and then create the candidate rows dynamically. Candidate rows retain the existing schema, checkpoint/resume behavior, one-filter increments, accuracy evaluation, and selected latency probes.

Automatic discovery will be deterministic: layer names are processed in model order and only layers returned by the existing safety adapter are included. Explicit layer lists and widths remain supported for existing experiments.

## Stop and failure behavior

For a given layer, mAP50-95 below 0.30 marks the current point as measured and skips only that layer's remaining filter removals. Other layers continue. Failed candidates are recorded with the existing error status and do not stop other layers.

## Verification

Unit tests will cover automatic planning from safe layers and widths, the 0.30 early-stop behavior, compatibility with existing explicit layer configurations, and stable resume artifacts. No full RTX sweep is run during implementation.
