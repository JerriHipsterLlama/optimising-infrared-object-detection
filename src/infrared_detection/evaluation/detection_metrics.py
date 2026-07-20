"""Common detection evaluation and metric normalization."""

from __future__ import annotations

from typing import Any, Mapping


CORE_METRICS = ("map50", "map50_95", "precision", "recall")


def _metric_value(metrics: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in metrics and metrics[key] is not None:
            value = metrics[key]
            return float(value.item() if hasattr(value, "item") else value)
    return None


def summarize_detection_metrics(results: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize evaluator output into the project-wide metric schema."""

    raw = results.get("metrics", results)
    if not isinstance(raw, Mapping):
        raise TypeError("Detection metrics must be a mapping.")
    summary = {
        "map50": _metric_value(raw, "map50", "mAP50", "metrics/mAP50(B)"),
        "map50_95": _metric_value(raw, "map50_95", "mAP50-95", "metrics/mAP50-95(B)"),
        "precision": _metric_value(raw, "precision", "precision(B)"),
        "recall": _metric_value(raw, "recall", "recall(B)"),
        "per_class_ap": dict(sorted((results.get("per_class_ap") or {}).items())),
    }
    for key in CORE_METRICS:
        if summary[key] is None:
            raise ValueError(f"Missing required detection metric: {key}")
    for key in ("image_count", "inference_time_ms", "device", "split"):
        if key in results:
            summary[key] = results[key]
    return summary


def evaluate_yolo(model: Any, data_yaml: str, split: str, imgsz: int, device: str | int, conf: float, iou: float) -> dict[str, Any]:
    """Run Ultralytics validation and normalize its box metrics."""

    results = model.val(data=data_yaml, split=split, imgsz=imgsz, device=device, conf=conf, iou=iou, verbose=False)
    box = results.box
    per_class = {}
    names = getattr(results, "names", {}) or {}
    class_maps = getattr(box, "maps", [])
    for class_id, value in enumerate(class_maps):
        per_class[str(names.get(class_id, class_id))] = float(value)
    return summarize_detection_metrics(
        {
            "metrics": {
                "map50": box.map50,
                "map50_95": box.map,
                "precision": box.mp,
                "recall": box.mr,
            },
            "per_class_ap": per_class,
            "device": str(device),
            "split": split,
        }
    )

