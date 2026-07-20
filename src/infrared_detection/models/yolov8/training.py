"""
YOLOv8n Training Script for CAMEL Infrared Dataset

Usage:
    python apps/train.py yolov8 --config configs/models/yolov8.yaml
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from infrared_detection.common import load_config


REPO_ROOT = Path(__file__).resolve().parents[4]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".npy"}


def count_dataset_images(image_dir: str | Path) -> int:
    """Count unique image stems across supported image representations."""

    directory = Path(image_dir)
    stems = {
        path.stem
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    return len(stems)


def resolve_dataset_yaml(dataset_yaml: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve a dataset YAML independently of the process working directory."""

    path = Path(dataset_yaml)
    resolved = path if path.is_absolute() else (repo_root / path)
    resolved = resolved.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {resolved}")
    return resolved


def resolve_training_device(device: str | int | None) -> str | int:
    """Validate an explicitly requested CUDA device instead of silently falling back."""

    if device is None:
        return 0 if torch.cuda.is_available() else "cpu"
    if isinstance(device, str) and device.lower() == "cpu":
        return "cpu"
    try:
        device_index = int(device)
    except (TypeError, ValueError) as exc:
        raise ValueError("Device must be 'cpu' or a CUDA device index such as 0.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {device_index} requested, but this PyTorch installation has no CUDA support."
        )
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device_index} is unavailable.")
    return device_index


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_yolov8(
    epochs: int = None,
    batch_size: int = None,
    img_size: int = None,
    resume: bool = False,
    config_path: str = 'configs/models/yolov8.yaml',
    device: str | int | None = None,
    seed: int | None = None,
    data: str | Path | None = None,
    project: str | Path | None = None,
    name: str = "train",
    dry_run: bool = False,
) -> Path:
    """
    Train YOLOv8n on CAMEL infrared dataset.

    Args:
        epochs (int): Number of epochs. If None, uses config value.
        batch_size (int): Batch size. If None, uses config value.
        img_size (int): Image size. If None, uses config value.
        resume (bool): Resume from last training. Default: False
        config_path (str): Path to config YAML file
    """
    config_path = str(resolve_dataset_yaml(config_path, REPO_ROOT)) if not Path(config_path).is_absolute() else config_path
    config = load_config(config_path)

    # Override config with CLI args if provided
    epochs = config['training']['epochs'] if epochs is None else epochs
    batch_size = config['training']['batch_size'] if batch_size is None else batch_size
    img_size = config['model']['img_size'] if img_size is None else img_size
    resolved_device = resolve_training_device(config['model']['device'] if device is None else device)
    resolved_seed = config.get('training', {}).get('seed', 7) if seed is None else seed
    seed_everything(int(resolved_seed))

    # Setup absolute checkpoint directory to avoid runs/detect/ prefix
    checkpoint_dir = (Path(project).resolve() if project is not None else Path(config['checkpoint']['resume_from']).resolve())

    # Dataset path
    dataset_yaml = resolve_dataset_yaml(
        data if data is not None else Path(config['data']['dataset_path']) / 'camel.yaml',
        REPO_ROOT,
    )

    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {dataset_yaml}\n"
            f"Make sure you have data/camel/camel.yaml configured"
        )

    # Verify dataset integrity - count images and labels
    dataset_root = Path(config['data']['dataset_path'])
    train_images_dir = dataset_root / 'images' / 'train'
    train_labels_dir = dataset_root / 'labels' / 'train'
    val_images_dir = dataset_root / 'images' / 'val'
    val_labels_dir = dataset_root / 'labels' / 'val'

    # Count files
    train_images_count = count_dataset_images(train_images_dir) if train_images_dir.exists() else 0
    train_labels = list(train_labels_dir.glob('*.txt')) if train_labels_dir.exists() else []
    val_images_count = count_dataset_images(val_images_dir) if val_images_dir.exists() else 0
    val_labels = list(val_labels_dir.glob('*.txt')) if val_labels_dir.exists() else []

    print(f"\n{'='*60}")
    print(f"DATASET VERIFICATION")
    print(f"{'='*60}")
    print(f"Training Set:")
    print(f"  Images: {train_images_count}")
    print(f"  Labels: {len(train_labels)}")
    print(f"  Match: {'✓' if train_images_count == len(train_labels) else '✗ MISMATCH!'}")
    print(f"\nValidation Set:")
    print(f"  Images: {val_images_count}")
    print(f"  Labels: {len(val_labels)}")
    print(f"  Match: {'✓' if val_images_count == len(val_labels) else '✗ MISMATCH!'}")
    print(f"{'='*60}\n")

    if train_images_count != len(train_labels):
        print(f"WARNING: Training images ({train_images_count}) != labels ({len(train_labels)})")
    if val_images_count != len(val_labels):
        print(f"WARNING: Validation images ({val_images_count}) != labels ({len(val_labels)})")

    print(f"Training YOLOv8n on infrared dataset...")
    print(f"  Config file: {config_path}")
    print(f"  Dataset: {dataset_yaml}")
    print(f"  Image size: {img_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Expected batches per epoch: {train_images_count // batch_size}")
    print(f"  Device: {resolved_device}")
    print(f"  Seed: {resolved_seed}")

    if dry_run:
        return checkpoint_dir / name

    # Initialize model only after dry-run validation has completed.
    model = YOLO(config['model']['name'])

    # Train with settings from config
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=resolved_device,
        patience=config['training']['patience'],
        save=True,
        save_period=config['checkpoint']['save_interval'],
        project=str(checkpoint_dir),  # Use absolute path to avoid runs/detect/ prefix
        name=name,
        resume=resume,
        pretrained=config['model']['pretrained'],

        # Thermal-optimized hyperparameters
        optimizer=config['training']['optimizer'],
        lr0=config['training']['learning_rate'],
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay'],
        warmup_epochs=config['training']['warmup_epochs'],

        # Augmentation
        degrees=config['augmentation']['degrees'],

        # Data loading
        workers=config['training']['num_workers'],  # Windows compatibility
        close_mosaic=15,

        verbose=True,
        compile=False,  # Disable compilation for better compatibility and debugging
        seed=int(resolved_seed),
    )

    print(f"\nTraining completed!")
    print(f"  Results saved to: {checkpoint_dir}/{name}")

    # Export to ONNX if configured
    if config['export']['onnx']:
        print(f"\nExporting to ONNX...")
        export_path = model.export(
            format='onnx',
            imgsz=img_size,
            opset=config['export']['onnx_opset'],
        )
        print(f"ONNX model exported: {export_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Train YOLOv8n on CAMEL infrared dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python apps/train.py yolov8                                    # Use config defaults
  python apps/train.py yolov8 --epochs 50 --batch-size 16        # Override specific params
  python apps/train.py yolov8 --resume                            # Resume from last checkpoint
  python apps/train.py yolov8 --config configs/models/yolov8.yaml --epochs 100
        ''',
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/models/yolov8.yaml',
        help='Path to config YAML file (default: configs/models/yolov8.yaml)',
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)',
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)',
    )

    parser.add_argument(
        '--img-size',
        type=int,
        default=None,
        help='Image size (overrides config)',
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from last checkpoint',
    )
    parser.add_argument('--device', default=None, help="CUDA device index or 'cpu'.")
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducible training.')
    parser.add_argument('--data', default=None, help='Dataset YAML path.')
    parser.add_argument('--project', default=None, help='Output project directory.')
    parser.add_argument('--name', default='train', help='Experiment run name.')
    parser.add_argument('--dry-run', action='store_true', help='Print resolved settings without training.')

    args = parser.parse_args()

    train_yolov8(
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        resume=args.resume,
        config_path=args.config,
        device=args.device,
        seed=args.seed,
        data=args.data,
        project=args.project,
        name=args.name,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    main()
