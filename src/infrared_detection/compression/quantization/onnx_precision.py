"""Precision conversion for exported ONNX models."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class CalibrationImageReader:
    """Stream deterministic, letterboxed calibration batches for ONNX Runtime."""

    def __init__(
        self,
        image_dir: str | Path,
        *,
        input_name: str,
        image_size: tuple[int, int],
        sample_limit: int | None = None,
        batch_size: int = 1,
        calibration_split: str | None = None,
    ) -> None:
        self.image_dir = Path(image_dir).resolve()
        if calibration_split is not None and calibration_split.strip().lower() == "test":
            raise ValueError("INT8 calibration must not use the test split")
        if "test" in {part.lower() for part in self.image_dir.parts}:
            raise ValueError(f"INT8 calibration path resolves to a test split: {self.image_dir}")
        if not input_name:
            raise ValueError("ONNX calibration input name must not be empty")
        if len(image_size) != 2 or any(int(value) <= 0 for value in image_size):
            raise ValueError("Calibration image size must be positive (height, width)")
        if batch_size != 1:
            raise ValueError("Calibration batch_size must be 1 for the exported fixed-batch YOLO graph")
        if sample_limit is not None and sample_limit <= 0:
            raise ValueError("Calibration sample limit must be positive")

        self.input_name = input_name
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.batch_size = batch_size
        self.image_paths = sorted(
            path for path in self.image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        ) if self.image_dir.is_dir() else []
        if sample_limit is not None:
            self.image_paths = self.image_paths[:sample_limit]
        self.sample_count = len(self.image_paths)
        self.preprocessing = {
            "resize": "letterbox",
            "color": "BGR to RGB",
            "layout": "NCHW",
            "dtype": "float32",
            "normalization": "divide by 255",
            "image_size": list(self.image_size),
            "batch_size": batch_size,
        }
        self._index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._index >= self.sample_count:
            return None

        import cv2

        images = []
        for path in self.image_paths[self._index : self._index + self.batch_size]:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read calibration image: {path}")
            images.append(self._letterbox(image))
        self._index += len(images)
        return {self.input_name: np.stack(images, axis=0)}

    def rewind(self) -> None:
        self._index = 0

    def _letterbox(self, image: np.ndarray) -> np.ndarray:
        import cv2

        target_height, target_width = self.image_size
        height, width = image.shape[:2]
        scale = min(target_height / height, target_width / width)
        resized_width = round(width * scale)
        resized_height = round(height * scale)
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        pad_width = target_width - resized_width
        pad_height = target_height - resized_height
        left = round(pad_width / 2 - 0.1)
        right = round(pad_width / 2 + 0.1)
        top = round(pad_height / 2 - 0.1)
        bottom = round(pad_height / 2 + 0.1)
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0


def calibration_reader_from_onnx(
    source: str | Path,
    image_dir: str | Path,
    *,
    sample_limit: int | None = None,
    batch_size: int = 1,
    calibration_split: str | None = None,
) -> CalibrationImageReader:
    """Build a reader using the exported graph's input name and spatial size."""

    import onnx

    model = onnx.load(str(source), load_external_data=False)
    initializers = {initializer.name for initializer in model.graph.initializer}
    inputs = [value for value in model.graph.input if value.name not in initializers]
    if len(inputs) != 1:
        raise ValueError(f"Expected one image input in ONNX model, found {len(inputs)}")
    dimensions = inputs[0].type.tensor_type.shape.dim
    if len(dimensions) != 4:
        raise ValueError("INT8 calibration requires a 4D NCHW ONNX image input")
    height, width = dimensions[2].dim_value, dimensions[3].dim_value
    if not height or not width:
        raise ValueError("INT8 calibration requires static ONNX input height and width")
    return CalibrationImageReader(
        image_dir,
        input_name=inputs[0].name,
        image_size=(height, width),
        sample_limit=sample_limit,
        batch_size=batch_size,
        calibration_split=calibration_split,
    )


def prepare_precision_onnx(
    source: str | Path,
    output: str | Path,
    precision: str,
    *,
    calibration_reader: Any | None = None,
    calibration_method: str = "entropy",
) -> dict[str, Any]:
    """Write an FP32, FP16, or INT8 ONNX graph and conversion metadata."""

    source = Path(source)
    output = Path(output)
    if not source.is_file():
        raise FileNotFoundError(f"Source ONNX model does not exist: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)

    if precision == "fp32":
        shutil.copy2(source, output)
        return {
            "converter": "copy",
            "precision": precision,
            "input_path": str(source.resolve()),
            "output_path": str(output.resolve()),
        }
    if precision == "fp16":
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "modelopt.onnx.autocast",
                "--onnx_path",
                str(source),
                "--output_path",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if not output.is_file():
            raise FileNotFoundError(f"ModelOpt FP16 conversion did not write output: {output}\n{result.stdout}\n{result.stderr}")
        return {
            "converter": "modelopt.onnx.autocast",
            "precision": precision,
            "input_path": str(source.resolve()),
            "output_path": str(output.resolve()),
            "modelopt_output": {"stdout": result.stdout, "stderr": result.stderr},
        }
    if precision != "int8":
        raise ValueError(f"Unsupported ONNX precision: {precision}")
    if calibration_method not in {"max", "entropy"}:
        raise ValueError("calibration_method must be 'max' or 'entropy'")
    if calibration_reader is None:
        raise ValueError("INT8 conversion requires a calibration reader")
    sample_count = int(getattr(calibration_reader, "sample_count", 0))
    if sample_count <= 0:
        raise ValueError("INT8 calibration dataset is empty")

    from modelopt.onnx.quantization import quantize

    modelopt_result = quantize(
        str(source),
        quantize_mode="int8",
        calibration_data_reader=calibration_reader,
        calibration_method=calibration_method,
        output_path=str(output),
    )
    if not output.is_file():
        raise FileNotFoundError(f"ModelOpt INT8 conversion did not write output: {output}")
    return {
        "converter": "modelopt.onnx.quantization.quantize",
        "precision": precision,
        "input_path": str(source.resolve()),
        "output_path": str(output.resolve()),
        "calibration_method": calibration_method,
        "calibration_source": str(getattr(calibration_reader, "image_dir", "")),
        "calibration_sample_count": sample_count,
        "calibration_preprocessing": getattr(calibration_reader, "preprocessing", None),
        "modelopt_output": repr(modelopt_result),
    }
