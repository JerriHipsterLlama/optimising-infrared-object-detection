"""Machine-readable experiment artifact writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_experiment_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    _write_json(path, manifest)


def write_metrics(path: str | Path, metrics: Mapping[str, Any]) -> None:
    _write_json(path, metrics)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    if isinstance(value, str):
        return value.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n")
    return value


def write_metrics_csv(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: _csv_value(row.get(key)) for key in keys} for row in rows)
