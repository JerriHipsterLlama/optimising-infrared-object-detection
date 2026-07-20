import json

from infrared_detection.evaluation.artifacts import write_experiment_manifest, write_metrics
from infrared_detection.evaluation.detection_metrics import summarize_detection_metrics


def test_summarize_detection_metrics_has_stable_core_schema():
    summary = summarize_detection_metrics(
        {
            "metrics": {"map50": 0.7, "map50_95": 0.4, "precision": 0.8, "recall": 0.6},
            "per_class_ap": {"dog": 0.3, "person": 0.5},
        }
    )

    assert list(summary)[:4] == ["map50", "map50_95", "precision", "recall"]
    assert summary["per_class_ap"] == {"dog": 0.3, "person": 0.5}


def test_artifacts_are_json_serializable_and_write_expected_files(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    metrics_path = tmp_path / "metrics.json"

    write_experiment_manifest(manifest_path, {"variant": "dense_fp32", "seed": 7})
    write_metrics(metrics_path, {"map50_95": 0.4, "latency_p50_ms": 12.0})

    assert json.loads(manifest_path.read_text(encoding="utf-8"))["variant"] == "dense_fp32"
    assert json.loads(metrics_path.read_text(encoding="utf-8"))["map50_95"] == 0.4
