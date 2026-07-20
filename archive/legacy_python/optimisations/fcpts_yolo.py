"""Run FCPTS calibration on a YOLOv8 checkpoint."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO
from torch.func import functional_call

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.python.optimisations.Pruning.fcpts_calibration import (  # noqa: E402
    CalibrationRunner,
    calibrate_model,
    finalize_and_export,
)


class CAMELCalibrationDataset(Dataset):
    """Representative grayscale images used for post-training calibration."""

    def __init__(self, image_dir: str | Path, limit: int | None = None, image_size: int = 320):
        self.image_dir = Path(image_dir)
        self.image_size = int(image_size)
        candidates = sorted(
            (path for path in self.image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".npy"}),
            key=lambda path: (path.stem, 0 if path.suffix.lower() != ".npy" else 1),
        )
        selected = {}
        for path in candidates:
            selected.setdefault(path.stem, path)
        self.paths = list(selected.values())[:limit] if limit is not None else list(selected.values())

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.paths[idx]
        if path.suffix.lower() == ".npy":
            image = np.load(path)
            if image.ndim == 3:
                image = image[..., 0]
        else:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read calibration image: {path}")
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).unsqueeze(0)
        return tensor


def _flatten_first_tensor(output: Any):
    if torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        for value in output.values():
            tensor = _flatten_first_tensor(value)
            if tensor is not None:
                return tensor
        return None
    if isinstance(output, (list, tuple)):
        for value in output:
            tensor = _flatten_first_tensor(value)
            if tensor is not None:
                return tensor
        return None
    return None


def yolo_feature_adapter(model, batch, targets=None, params_override=None):
    del targets
    if batch.dim() == 4 and batch.shape[1] == 1:
        batch = batch.repeat(1, 3, 1, 1)
    model.eval()
    output = model(batch) if params_override is None else functional_call(model, params_override, (batch,))
    tensor = _flatten_first_tensor(output)
    if tensor is None:
        raise RuntimeError("YOLO adapter could not find tensor output.")
    return [tensor]


def load_yolo_state_dict_into_model(model, state_dict_path: str | Path) -> YOLO:
    """Load a finalized FCPTS state dictionary into a YOLO wrapper."""

    yolo = model if isinstance(model, YOLO) else YOLO(str(model))
    payload = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state_dict, dict):
        raise TypeError("FCPTS checkpoint must contain a state dictionary.")
    incompatible = yolo.model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(f"FCPTS checkpoint is missing model keys: {incompatible.missing_keys[:5]}")
    yolo.model.eval()
    return yolo


def _load_yaml_config(config_path: Path | None) -> Dict[str, Any]:
    if config_path is None:
        return {}
    config_abs = (REPO_ROOT / config_path).resolve() if not config_path.is_absolute() else config_path
    with open(config_abs, "r", encoding="utf-8") as handle:
        parsed = yaml.safe_load(handle) or {}
    if not isinstance(parsed, dict):
        raise ValueError("Config file must parse to a dictionary.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FCPTS calibration on a YOLOv8 checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to YOLO .pt checkpoint.")
    parser.add_argument("--config", type=Path, default=None, help="YAML config path for training/runtime defaults.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models\\baseline\\yolov8\\train\\weights\\best_fcpts.pt"),
        help="Path for finalized sparse checkpoint (state_dict).",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("runs\\fcpts_yolov8.csv"),
        help="CSV path for iteration metrics.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Calibration epochs (overrides config).")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config).")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Device (overrides config).")
    parser.add_argument("--batch-size", type=int, default=1, help="Calibration batch size.")
    parser.add_argument("--samples", type=int, default=4, help="Number of random calibration samples.")
    parser.add_argument("--calibration-dir", type=Path, default=Path("data/camel/images/train"), help="Representative image directory.")
    parser.add_argument("--image-size", type=int, default=320, help="Square random input size.")
    parser.add_argument("--target-sparsity", type=float, default=0.5, help="Target global sparsity r0.")
    parser.add_argument("--lambda-control", type=float, default=1.0, help="Control loss weight.")
    parser.add_argument("--bandwidth", type=float, default=0.1, help="KDE bandwidth.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved settings and exit.")
    return parser.parse_args()


def _resolve_hparams(args: argparse.Namespace, config: Dict[str, Any]) -> tuple[int, float, str]:
    training_cfg = config.get("training", {}) if isinstance(config.get("training", {}), dict) else {}
    runtime_cfg = config.get("runtime", {}) if isinstance(config.get("runtime", {}), dict) else {}

    epochs = args.epochs if args.epochs is not None else int(training_cfg.get("epochs", 1))
    lr = args.lr if args.lr is not None else float(training_cfg.get("lr", 1e-4))
    device = args.device if args.device is not None else str(runtime_cfg.get("device", "cpu"))
    return epochs, lr, device


def main() -> None:
    args = parse_args()
    config = _load_yaml_config(args.config)
    epochs, lr, device = _resolve_hparams(args, config)

    checkpoint = (REPO_ROOT / args.checkpoint).resolve() if not args.checkpoint.is_absolute() else args.checkpoint
    output_path = (REPO_ROOT / args.output).resolve() if not args.output.is_absolute() else args.output
    metrics_csv = (REPO_ROOT / args.metrics_csv).resolve() if not args.metrics_csv.is_absolute() else args.metrics_csv

    if args.dry_run:
        print(f"checkpoint={checkpoint}")
        print(f"epochs={epochs}")
        print(f"lr={lr}")
        print(f"device={device}")
        return

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    dense_model = YOLO(str(checkpoint)).model.eval()
    sparse_model = copy.deepcopy(dense_model)
    for param in sparse_model.parameters():
        param.requires_grad_(True)

    calibration_dir = (REPO_ROOT / args.calibration_dir).resolve() if not args.calibration_dir.is_absolute() else args.calibration_dir
    dataloader = DataLoader(
        CAMELCalibrationDataset(image_dir=calibration_dir, limit=args.samples, image_size=args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
    )
    runner = CalibrationRunner(
        dense_model=dense_model,
        sparse_model=sparse_model,
        output_adapter=yolo_feature_adapter,
        target_sparsity=args.target_sparsity,
        lambda_control=args.lambda_control,
        bandwidth=args.bandwidth,
    )
    runner.sparse_model.eval()

    calibrate_model(
        runner=runner,
        dataloader=dataloader,
        epochs=epochs,
        lr=lr,
        device=torch.device(device),
        metrics_csv_path=metrics_csv,
    )
    finalized = finalize_and_export(
        sparse_model=sparse_model,
        threshold_registry=runner.threshold_registry(),
        bandwidth=args.bandwidth,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(finalized.state_dict(), output_path)
    print(f"Saved sparse checkpoint state_dict to: {output_path}")
    print(f"Saved calibration metrics CSV to: {metrics_csv}")


if __name__ == "__main__":
    main()
