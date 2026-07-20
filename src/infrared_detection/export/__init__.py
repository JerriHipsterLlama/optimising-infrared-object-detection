"""Model export adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def export_yolo(model_path: str | Path, format: str = "onnx", **kwargs: Any) -> Any:
    """Export a YOLO model through Ultralytics without coupling evaluation code to it."""

    from ultralytics import YOLO

    return YOLO(str(model_path)).export(format=format, **kwargs)
