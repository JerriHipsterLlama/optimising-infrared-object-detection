# Task 2: Build and validate the selected structured-pruned checkpoint

Build the selected structured-pruning artifact without modifying the active filterwise run.

Files:
- Modify `src/infrared_detection/evaluation/compression_matrix.py`
- Modify `src/infrared_detection/compression/pruning/cluster_selection.py` only if needed for explicit cluster planning
- Test `tests/unit/test_compression_matrix_workflow.py`
- Test `tests/unit/test_cluster_selection.py`

Required interfaces:
- `select_filterwise_candidate(manifest_path: Path, layer: str, filters_removed: int) -> Mapping[str, Any]` (evidence lookup only)
- `build_pruned_checkpoint(config: Mapping[str, Any], output_dir: Path) -> Path`

Requirements:
- Read the explicit filterwise manifest path and evidence layer/removal count from Task 1 configuration, and use the evidence row only to validate the cluster-size choice.
- Reject missing, failed, or incomplete filterwise evidence rows before pruning, but do not prune the evidence checkpoint again.
- Start actual pruning from the dense `model.checkpoint` configured in YAML.
- Read `candidate_layers`, `cluster_size`, `prune_ratios`, `allowed_map50_95_drop`, and `importance: minimum_weight` explicitly from YAML.
- Build a deterministic selected grid ratio through an explicit `requested_ratio` parameter; Task 4 selects the highest ratio whose measured mAP50_95 drop is within `allowed_map50_95_drop`.
- For every configured candidate layer, compute Minimum-Weight channel importance, form complete low-importance clusters of exactly `cluster_size`, and remove the selected ratio rounded down to complete clusters through the existing dependency graph.
- Save `pruned.pt`, reload it, and record parameter-count and serialized-size changes.
- Preserve the global constraints: official precisions remain exactly FP32/FP16/INT8; active filterwise outputs remain read-only; no KD.

Tests must cover:

```python
def test_select_filterwise_candidate_requires_matching_layer_and_count(tmp_path):
    manifest = tmp_path / "filterwise.json"
    manifest.write_text(json.dumps({"rows": [{"layer": "model.2.cv2.conv", "filters_removed": 8, "status": "screened_in", "checkpoint_path": "candidate.pt"}]}))
    row = select_filterwise_candidate(manifest, "model.2.cv2.conv", 8)
    assert row["checkpoint_path"] == "candidate.pt"

def test_select_filterwise_candidate_rejects_failed_row(tmp_path):
    manifest = tmp_path / "filterwise.json"
    manifest.write_text(json.dumps({"rows": [{"layer": "model.2.cv2.conv", "filters_removed": 8, "status": "failed"}]}))
    with pytest.raises(ValueError, match="not usable"):
        select_filterwise_candidate(manifest, "model.2.cv2.conv", 8)
```

Run focused tests with:
`.venv\Scripts\python.exe -m pytest tests\unit\test_compression_matrix_workflow.py tests\unit\test_cluster_selection.py -q`

Commit the implementation and append the full report to:
`.superpowers\sdd\2026-07-27-rtx-compression-pipeline\task-2-report.md`

The report must include status, commit hashes, tests run and output, and concerns.
