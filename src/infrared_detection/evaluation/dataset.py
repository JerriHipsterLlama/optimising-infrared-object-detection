"""Dataset utilities shared by calibration and evaluation workflows."""

from __future__ import annotations

from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".npy"}


def list_image_files(image_dir: str | Path) -> list[Path]:
    directory = Path(image_dir)
    paths = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(paths, key=lambda path: (path.stem, path.suffix.lower()))

