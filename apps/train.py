"""Training command-line entry point for supported model families.

The source checkout is supported as an executable development entry point, so
this module adds the repository's ``src`` directory before importing trainers.
Installed package entry points do not require this bootstrap.
"""

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"


def _bootstrap_source_imports() -> None:
    """Make ``src`` imports available when this file is run from a checkout."""

    source_root = str(SOURCE_ROOT)
    if source_root not in sys.path:
        sys.path.insert(0, source_root)


_bootstrap_source_imports()


def _train_yolov8(args: argparse.Namespace) -> None:
    from infrared_detection.models.yolov8.training import train_yolov8

    train_yolov8(
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        resume=args.resume,
        device=args.device,
        seed=args.seed,
        data=args.data,
        project=args.project,
        name=args.name,
        dry_run=args.dry_run,
    )


def _train_faster_rcnn(args: argparse.Namespace) -> None:
    from infrared_detection.models.faster_rcnn.training import train_faster_rcnn

    train_faster_rcnn(
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        resume=args.resume,
        checkpoint_path=args.checkpoint,
        name=args.name,
        no_augment=args.no_augment,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an infrared-detection model.")
    subparsers = parser.add_subparsers(dest="model_family", required=True)

    yolov8_parser = subparsers.add_parser("yolov8", help="Train a YOLOv8 model.")
    yolov8_parser.add_argument(
        "--config",
        default="configs/models/yolov8.yaml",
        help="Path to the YOLOv8 configuration file.",
    )
    yolov8_parser.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    yolov8_parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size."
    )
    yolov8_parser.add_argument(
        "--img-size", type=int, default=None, help="Override image size."
    )
    yolov8_parser.add_argument("--resume", action="store_true", help="Resume training.")
    yolov8_parser.add_argument("--device", default=None, help="CUDA device index or 'cpu'.")
    yolov8_parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    yolov8_parser.add_argument("--data", default=None, help="Dataset YAML path.")
    yolov8_parser.add_argument("--project", default=None, help="Output project directory.")
    yolov8_parser.add_argument("--name", default="train", help="Experiment run name.")
    yolov8_parser.add_argument(
        "--dry-run", action="store_true", help="Validate resolved settings without training."
    )
    yolov8_parser.set_defaults(handler=_train_yolov8)

    faster_rcnn_parser = subparsers.add_parser(
        "faster-rcnn", help="Train a Faster R-CNN model."
    )
    faster_rcnn_parser.add_argument(
        "--config",
        default="configs/models/faster_rcnn.yaml",
        help="Path to the Faster R-CNN configuration file.",
    )
    faster_rcnn_parser.add_argument(
        "--epochs", type=int, default=None, help="Override epochs."
    )
    faster_rcnn_parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size."
    )
    faster_rcnn_parser.add_argument("--resume", action="store_true", help="Resume training.")
    faster_rcnn_parser.add_argument(
        "--checkpoint", default=None, help="Checkpoint path to resume from."
    )
    faster_rcnn_parser.add_argument("--name", default="train", help="Experiment run name.")
    faster_rcnn_parser.add_argument(
        "--no-augment", action="store_true", help="Disable training augmentations."
    )
    faster_rcnn_parser.set_defaults(handler=_train_faster_rcnn)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
