"""Export trained models to deployment formats."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a YOLO checkpoint to ONNX or another Ultralytics format.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--format", default="onnx")
    parser.add_argument("--imgsz", type=int, default=None)
    args = parser.parse_args()
    from infrared_detection.export import export_yolo

    kwargs = {} if args.imgsz is None else {"imgsz": args.imgsz}
    export_yolo(args.model, format=args.format, **kwargs)


if __name__ == "__main__":
    main()
