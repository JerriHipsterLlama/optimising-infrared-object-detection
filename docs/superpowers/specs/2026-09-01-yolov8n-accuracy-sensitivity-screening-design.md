# YOLOv8n Accuracy Sensitivity Screening Design

## Purpose

Add a standalone workstation workflow that measures the immediate validation-accuracy sensitivity of a trained YOLOv8n detector to independent structural channel pruning. Each pruning unit is evaluated at requested ratios `0.125`, `0.25`, `0.375`, and `0.50` without fine-tuning. Every candidate starts from a new load of the original checkpoint, so pruning never accumulates across units or ratios.

This workflow is distinct from the existing `single_layer_performance_screening` workflow. The existing workflow performs cumulative filter-by-filter pruning and latency measurement. It remains unchanged. The new workflow performs four independent ratio probes per unit and produces accuracy-degradation and AUC sensitivity results.

## Command and Files

The standalone command is:

```powershell
.venv\Scripts\python.exe apps/yolov8_accuracy_sensitivity.py --config configs/experiments/yolov8n_accuracy_sensitivity.yaml
```

The implementation uses these boundaries:

- `apps/yolov8_accuracy_sensitivity.py`: thin CLI and progress reporting.
- `src/infrared_detection/compression/pruning/unit_discovery.py`: region classification, dependency-group inspection, and structural validation.
- `src/infrared_detection/evaluation/accuracy_sensitivity.py`: configuration, orchestration, metric calculation, persistence, and resume behavior.
- `configs/experiments/yolov8n_accuracy_sensitivity.yaml`: checkpoint, validation, runtime, pruning, and output settings.
- `tests/unit/test_pruning_unit_discovery.py`: dependency classification and Detect-contract tests.
- `tests/unit/test_accuracy_sensitivity.py`: ratio planning, independence, metrics, AUC, CSV, failure, and resume tests.
- `tests/integration/test_yolov8n_accuracy_sensitivity_graph.py`: real YOLOv8n graph discovery and pruning smoke tests.

The output directory contains exactly the durable research artifacts:

```text
results.csv
unit_sensitivity_ranking.csv
manifest.json
```

No candidate checkpoint, ONNX file, TensorRT engine, or fine-tuned model is saved.

## Configuration

The YAML configuration contains:

- `model.checkpoint`: trained YOLOv8n checkpoint.
- `data.dataset_yaml`: validation dataset YAML.
- `data.split`: validation split.
- `validation`: arguments applied identically to baseline and candidate `YOLO.val` calls, including image size, batch size, device, precision, confidence threshold, IoU threshold, workers, and seed-related deterministic settings where supported.
- `pruning.ratios`: defaults to `[0.125, 0.25, 0.375, 0.50]`.
- `pruning.importance`: defaults to `l1`.
- `pruning.example_image_size`: a stride-compatible size used only for dependency tracing and structural forward validation.
- `experiment.output_dir`: artifact directory.

Unknown or duplicate ratios, ratios outside `(0, 1)`, a missing checkpoint or dataset YAML, a non-YOLOv8n model, and an unsupported importance criterion fail before baseline evaluation. The resolved validation argument mapping is constructed once and passed unchanged to every evaluation.

## Actual Graph Inspection

The design was checked against `models/checkpoints/yolov8/train3/weights/best.pt` with PyTorch `2.9.1+cu128`, Ultralytics `8.4.7`, and Torch-Pruning `1.6.0`. The model has 23 top-level modules and 64 `Conv2d` modules.

At all four requested ratios, dependency-group construction, structural pruning, and forward validation identified:

- 39 independently valid channel-producing roots.
- 18 C2f split or residual-ending roots whose groups remove output channels from other meaningful convolution units. These are `GROUPED`, not independent candidates.
- Six Detect prediction-output convolutions whose pruning groups are rejected because regression and class output widths are fixed.
- One DFL convolution with a single output channel, which cannot remove a channel while retaining a valid layer.

The 39 valid roots consist of 27 backbone/neck convolutions and 12 hidden Detect-tower convolutions. Dynamic discovery, rather than this observed count or a name allowlist, remains authoritative at runtime.

## Architectural Regions

Every convolution is assigned one of three regions from the loaded model structure:

- `backbone`: top-level modules before the first feature-pyramid upsample or concatenation.
- `neck`: feature-pyramid modules beginning with the first upsample or concatenation and ending before `Detect`.
- `detect_head`: descendants of the `Detect` module.

The classifier derives boundaries from module types and parentage rather than fixed numeric layer indexes. If the structure cannot be classified unambiguously as YOLOv8n, discovery fails with a descriptive error.

## Candidate Enumeration and L1 Ranking

Discovery enumerates every `nn.Conv2d` in deterministic `named_modules()` order. A module with fewer than two output channels is recorded as `INVALID` with reason `insufficient_channels`.

For output channel `k`, the default importance score is true L1 filter magnitude:

\[
I_k = \sum_{c,h,w} |W_{k,c,h,w}|.
\]

Channels are stable-sorted by `(score, original_channel_index)` in ascending order. The scoring function is injected through an importance-criterion interface so another criterion can later replace L1 without changing discovery or orchestration.

For original output width `C` and requested ratio `r`, the requested prune count is:

\[
K = \operatorname{round}(C r).
\]

The configured YOLOv8n widths make all four default ratios integral. General configurations must satisfy `1 <= K < C`; otherwise the row is `INVALID`. The `K` least-important original channels are requested together in one Torch-Pruning group.

## Dependency-Group Audit

Every `(unit, ratio)` receives a new model load and a new Torch-Pruning dependency graph. Validation is repeated at every ratio; a unit-level probe is not reused as evidence for higher ratios.

Before pruning, the workflow serializes the full dependency group. Each operation records:

- resolved module name, or a stable generated identifier for an unnamed graph operation;
- module type;
- pruning handler/operation;
- affected channel indices;
- number of affected indices.

The detailed CSV stores `touched_modules` as a deterministic semicolon-separated summary and `dependency_group_json` as the complete JSON representation.

A dependency group may contain:

- output-channel removal from the selected root convolution;
- corresponding BatchNorm channel removal;
- downstream input-channel removal;
- mechanical graph operations for concatenation, split, reshape, elementwise activation, pooling, and upsampling.

If the dependency group removes output channels from any other independently meaningful `Conv2d` or `Linear` module, the candidate is `GROUPED`. The complete group is recorded, but it is not pruned or evaluated. This prevents Torch-Pruning propagation from silently turning one-unit sensitivity measurement into a coupled multi-unit experiment.

The selected root is the only meaningful channel-producing unit allowed to lose output channels in a `VALID` experiment. BatchNorm propagation and downstream input-channel changes are dependencies of that unit rather than additional pruning units.

## Structural and Detect Validation

For a dependency-accepted, non-grouped candidate, the workflow applies the group in memory and synchronizes physical channel metadata. It records:

- original root output channels;
- requested prune count;
- requested remaining channels (`C - K`);
- actual remaining channels read from the pruned weight tensor;
- requested ratio;
- achieved ratio `(C - actual_remaining_channels) / C`.

The candidate is `INVALID` if no physical root-channel reduction occurs, all root channels are removed, the actual reduction differs unexpectedly from the inspected group, or a forward pass fails.

The baseline Detect contract is captured from a forward pass before screening. The contract includes:

- number of detection scales;
- decoded prediction tensor rank and output-channel dimension;
- regression-box tensor rank and channel dimension;
- class-score tensor rank and class-channel dimension;
- `Detect.nc`, `Detect.reg_max`, `Detect.no`, and `Detect.nl`.

Every structurally pruned candidate must match all these values. Feature-map channel widths feeding Detect may change because that is the purpose of structural pruning, but the number of scales, prediction dimensionality, regression representation, class count, and output semantics may not. A mismatch is recorded as `SEMANTICS_CHANGED` and the candidate is not sent to dataset validation.

For the inspected four-class checkpoint the baseline contract is three scales, a rank-3 decoded tensor with eight outputs per location, a rank-3 regression tensor with 64 channels, a rank-3 class-score tensor with four channels, `nc=4`, `reg_max=16`, `no=68`, and `nl=3`.

## Independent Experiment Lifecycle

The baseline is loaded and evaluated once. Discovery may use that unmodified model, but it never mutates it.

Each `(unit, ratio)` then follows this lifecycle:

1. Load the original checkpoint from disk into a new YOLO wrapper.
2. Recompute the selected unit's L1 ranking from this dense copy.
3. Build a fresh dependency graph using a stride-compatible example tensor on the model's device and dtype.
4. Inspect and serialize the complete pruning group.
5. Mark and persist `GROUPED` or `INVALID` without evaluation when appropriate.
6. Apply an accepted group structurally in memory.
7. Verify requested and actual root-channel counts.
8. Run the full Detect-contract forward validation.
9. Validate the pruned model with the exact baseline dataset and validation arguments.
10. Calculate metrics, persist the row, release references, and clear accelerator cache where applicable.

There is no fine-tuning, retraining, calibration, or weight adaptation. The in-memory pruned candidate is discarded after validation. The next combination starts again at step 1.

## Accuracy Metrics and Point Sensitivity

Baseline and candidate rows record:

- `map50_95`;
- `map50`;
- `precision`;
- `recall`.

For baseline mAP50-95 `A0`, candidate accuracy `Ai,r`, and achieved ratio `r_hat_i,r`, the row records:

\[
\Delta A_{i,r}=A_0-A_{i,r}
\]

and point sensitivity:

\[
S_{i,r}=\frac{A_0-A_{i,r}}{\hat r_{i,r}}.
\]

Accuracy improvement is retained as a negative degradation and negative sensitivity; values are not clipped.

## Aggregate AUC Sensitivity

For each unit with four successful measurements, form the degradation curve from `(0, 0)` and the four points `(r_hat_i,r, A0 - Ai,r)`, sorted by achieved ratio. Duplicate or non-increasing achieved ratios invalidate aggregation.

The aggregate score is normalized trapezoidal AUC:

\[
S_i^{AUC}=\frac{1}{\hat r_{max}}\int_0^{\hat r_{max}}(A_0-A_i(r))\,dr.
\]

Higher scores indicate greater accuracy sensitivity. Negative areas remain negative. Units are ranked in descending AUC sensitivity with module name as the stable tie-breaker.

A unit missing any requested-ratio measurement because of `GROUPED`, `INVALID`, `SEMANTICS_CHANGED`, or runtime evaluation failure is included in the aggregate CSV with status `INCOMPLETE`, blank AUC and rank, and its successful-point count. This prevents incomplete curves from being compared with complete four-point curves.

## Detailed Result CSV

`results.csv` begins with one `BASELINE` row and then one deterministic row per enumerated `(Conv2d, ratio)` pair. This includes non-evaluated `GROUPED`, `INVALID`, and `SEMANTICS_CHANGED` rows so the discovery decision is auditable.

The leading fields are:

1. `candidate_id`
2. `status`
3. `pruning_unit`
4. `architectural_region`
5. `original_channels`
6. `requested_pruned_channels`
7. `requested_remaining_channels`
8. `actual_remaining_channels`
9. `requested_pruning_ratio`
10. `actual_pruning_ratio`
11. `map50_95`
12. `map50`
13. `precision`
14. `recall`
15. `map50_95_change_from_baseline`
16. `point_sensitivity`
17. `touched_modules`
18. `dependency_group_json`
19. `status_reason`
20. `error`

Provenance fields follow, including checkpoint, dataset, split, importance criterion, validation settings fingerprint, library versions, and elapsed validation time. `map50_95_change_from_baseline` uses candidate minus baseline for an intuitive signed change; `accuracy_degradation` is also stored explicitly as baseline minus candidate to match the sensitivity formula.

Rows are written atomically after each combination. Runtime failures record `ERROR` and continue when the next experiment can safely start from the dense checkpoint. Keyboard interruption writes the completed rows and exits.

## Aggregate Ranking CSV

`unit_sensitivity_ranking.csv` contains one row per meaningful candidate unit with:

- aggregate rank;
- pruning unit;
- architectural region;
- curve status;
- original channels;
- successful ratio count;
- maximum achieved ratio;
- normalized AUC sensitivity;
- mAP50-95 values and degradations at each requested ratio.

Only `COMPLETE` curves receive an aggregate rank. The file is regenerated atomically after every detailed-result update.

## Manifest and Resume

`manifest.json` records the resolved configuration, checkpoint and dataset identities, experiment fingerprint, baseline metrics, Detect contract, ordered candidate-unit inventory, library versions, start time, and completion state.

The experiment fingerprint includes checkpoint and dataset path/size/mtime, validation arguments, ratios, importance criterion/version, and relevant library versions. On restart, matching successful, grouped, or terminal-invalid rows are reused. An `ERROR` row is retried from a fresh checkpoint. A fingerprint mismatch fails rather than mixing results from different experiments.

Resume requires no pruned checkpoint because each candidate is independent.

## Error Handling

- Configuration and model-shape errors fail before baseline validation.
- Torch-Pruning group rejection produces `INVALID` with a dependency reason.
- Coupled meaningful output pruning produces `GROUPED` and never mutates or evaluates the candidate.
- Forward failure produces `INVALID` with the exception summary.
- Detect-contract mismatch produces `SEMANTICS_CHANGED` with expected and actual contracts.
- Dataset validation failure produces `ERROR`; the next combination may continue because it reloads the dense checkpoint.
- A baseline validation failure aborts the workflow because sensitivity cannot be calculated.

## Testing Strategy

All new behavior is implemented test-first.

Unit tests use small real PyTorch graphs to prove:

- L1 scores and stable ranking are correct.
- Requested prune counts and achieved ratios are correct.
- A sequential Conv/BN/Conv group is independent and records output, BN, and downstream input operations.
- A residual graph that couples another convolution's output is `GROUPED` and is not pruned.
- Classification is repeated independently for all four ratios.
- region discovery returns `backbone`, `neck`, and `detect_head` from structural boundaries.
- Detect hidden convolutions can remain valid while terminal prediction convolutions cannot change output semantics.
- every candidate load originates from the dense checkpoint and histories never accumulate.
- baseline and candidates receive equal validation argument mappings.
- point sensitivity preserves positive and negative values.
- normalized trapezoidal AUC includes `(0, 0)`, uses achieved ratios, and ranks descending.
- incomplete curves receive no AUC rank.
- detailed and aggregate CSV columns and JSON dependency serialization are deterministic.
- resume skips terminal/completed rows, retries `ERROR`, and rejects fingerprint changes.

The real-checkpoint integration test loads the supplied trained YOLOv8n model, confirms the three architectural regions, verifies representative backbone, C2f, neck/concatenation, residual-grouped, Detect-hidden, Detect-terminal, and DFL outcomes at all four ratios, and checks the complete Detect contract after valid structural pruning. It does not run the validation dataset, keeping automated tests bounded.

A manual workstation smoke run uses the trained checkpoint and validation YAML with one unit and one ratio before starting the full 156-valid-experiment screen.

## Success Criteria

The implementation is complete when:

- baseline validation and every successful candidate use an identical validation configuration;
- every `(unit, ratio)` starts from a newly loaded original checkpoint;
- every combination receives an independently constructed and recorded dependency group;
- no `VALID` candidate changes another meaningful unit's output width;
- physical structural pruning changes root output and dependent input/normalization tensors;
- every structurally accepted candidate preserves the complete Detect output contract;
- all valid hidden Detect convolutions remain eligible;
- all rows carry `backbone`, `neck`, or `detect_head` region tags;
- the detailed CSV contains requested and actual channel counts, ratios, metrics, degradation, point sensitivity, and full dependency-group data;
- the aggregate CSV contains normalized AUC rankings only for complete curves;
- no fine-tuning occurs and no pruned model is permanently stored;
- the focused unit and integration tests pass.
