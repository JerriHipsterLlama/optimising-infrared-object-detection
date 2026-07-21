# RTX Screening Mode Design

## Goal

Provide an explicit RTX-only screening run for the cluster-pruning experiment without allowing RTX measurements to determine the final Jetson Orin Nano winner.

## Behavior

`--screen-only` runs baseline validation and single-layer structural probes on `targets.rtx_screening_device`, records accuracy, serialized size, parameter reduction, and RTX latency, and marks all global candidates as skipped. It does not classify a final winner or write Orin hardware provenance.

Screening artifacts use a separate output directory so they cannot overwrite the authoritative Orin experiment. The existing default workflow remains Orin-gated.

## Testing

Unit tests verify that screen-only uses the RTX execution device and that global candidates are skipped. The CLI dry-run remains hardware-free.
