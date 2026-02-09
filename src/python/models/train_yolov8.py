"""
YOLOv8n Training Script for CAMEL Infrared Dataset

Usage:
    python scripts/train_yolov8.py
    python scripts/train_yolov8.py --resume
"""

import argparse
import yaml
from pathlib import Path

from ultralytics import YOLO

def load_config(config_path: str = 'configs/yolov8_config.yaml') -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_yolov8(
    epochs: int = None,
    batch_size: int = None,
    img_size: int = None,
    resume: bool = False,
    config_path: str = 'configs/yolov8_config.yaml',
):
    """
    Train YOLOv8n on CAMEL infrared dataset.
    
    Args:
        epochs (int): Number of epochs. If None, uses config value.
        batch_size (int): Batch size. If None, uses config value.
        img_size (int): Image size. If None, uses config value.
        resume (bool): Resume from last training. Default: False
        config_path (str): Path to config YAML file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Override config with CLI args if provided
    epochs = epochs or config['training']['epochs']
    batch_size = batch_size or config['training']['batch_size']
    img_size = img_size or config['model']['img_size']
    
    # Initialize model
    model = YOLO(config['model']['name'])
    
    # Dataset path
    dataset_yaml = Path(config['data']['dataset_path']) / 'camel.yaml'
    
    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {dataset_yaml}\n"
            f"Make sure you have data/camel/camel.yaml configured"
        )
    
    print(f"Training YOLOv8n on infrared dataset...")
    print(f"  Config file: {config_path}")
    print(f"  Dataset: {dataset_yaml}")
    print(f"  Image size: {img_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    
    # Train with settings from config
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=config['model']['device'],
        patience=config['training']['patience'],
        save=True,
        save_period=config['checkpoint']['save_interval'],
        project=config['checkpoint']['resume_from'],
        name='train',
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
        compile=True,
    )
    
    print(f"\nTraining completed!")
    print(f"  Results saved to: models/checkpoints/yolov8n/train")
    
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
  python scripts/train_yolov8.py                                  # Use config defaults
  python scripts/train_yolov8.py --epochs 50 --batch-size 16     # Override specific params
  python scripts/train_yolov8.py --resume                         # Resume from last checkpoint
  python scripts/train_yolov8.py --config configs/yolov8_config.yaml --epochs 100
        ''',
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/yolov8_config.yaml',
        help='Path to config YAML file (default: configs/yolov8_config.yaml)',
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
    
    args = parser.parse_args()
    
    train_yolov8(
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        resume=args.resume,
        config_path=args.config,
    )


if __name__ == '__main__':
    main()

