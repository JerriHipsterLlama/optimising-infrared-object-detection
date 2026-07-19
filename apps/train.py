"""Training command-line entry point for supported model families."""

import argparse


def _train_yolov8(args: argparse.Namespace) -> None:
    from infrared_detection.models.yolov8.training import train_yolov8

    train_yolov8(config_path=args.config)


def _train_faster_rcnn(args: argparse.Namespace) -> None:
    from infrared_detection.models.faster_rcnn.training import train_faster_rcnn

    train_faster_rcnn(config_path=args.config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an infrared-detection model.")
    subparsers = parser.add_subparsers(dest="model_family", required=True)

    yolov8_parser = subparsers.add_parser("yolov8", help="Train a YOLOv8 model.")
    yolov8_parser.add_argument(
        "--config",
        default="configs/yolov8_config.yaml",
        help="Path to the YOLOv8 configuration file.",
    )
    yolov8_parser.set_defaults(handler=_train_yolov8)

    faster_rcnn_parser = subparsers.add_parser(
        "faster-rcnn", help="Train a Faster R-CNN model."
    )
    faster_rcnn_parser.add_argument(
        "--config",
        default="configs/faster_rcnn_config.yaml",
        help="Path to the Faster R-CNN configuration file.",
    )
    faster_rcnn_parser.set_defaults(handler=_train_faster_rcnn)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
