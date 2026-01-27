"""
YOLOv8n Training Script for CAMEL Infrared Dataset

This script trains YOLOv8n on infrared images with built-in checkpointing,
logging, and metric tracking. Can resume from checkpoints and validates
at specified intervals.

Usage:
    python scripts/train_yolov8.py --config configs/yolov8_config.yaml
    python scripts/train_yolov8.py --config configs/yolov8_config.yaml --resume models/checkpoints/yolov8n_epoch_50.pt
    python scripts/train_yolov8.py --config configs/yolov8_config.yaml --epochs 100 --batch-size 16
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np

# Add src/python to path for imports FIRST
src_python_path = str(Path(__file__).parent.parent / "src" / "python")
if src_python_path not in sys.path:
    sys.path.insert(0, src_python_path)

# Now import YOLO - try different import paths
try:
    from ultralytics import YOLO
except ImportError:
    try:
        from ultralytics.yolo import detect
        # Fallback will work with native YOLO training
        YOLO = None
    except ImportError:
        raise ImportError("Could not import YOLO from ultralytics. Install with: pip install ultralytics")

# Import custom dataset utilities
from dataset.dataset import create_dataloaders

from tqdm import tqdm


class YOLOv8Trainer:
    """Trainer for YOLOv8n on CAMEL infrared dataset with checkpointing."""
    
    def __init__(self, config_path: str, resume_from: Optional[str] = None):
        """
        Initialize trainer with config.
        
        Args:
            config_path (str): Path to YAML config file
            resume_from (str): Optional path to checkpoint to resume from
        """
        self.config = self._load_config(config_path)
        self.device = torch.device(f"cuda:{self.config['model']['device']}" 
                                   if torch.cuda.is_available() else "cpu")
        
        # Setup logging
        self._setup_logging()
        self.logger.info(f"Using device: {self.device}")
        
        # Create checkpoint and log directories
        Path(self.config['checkpoint']['save_path']).mkdir(parents=True, exist_ok=True)
        Path(self.config['logging']['log_dir']).mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.start_epoch = 0
        self.best_metric = float('inf')
        self.metrics_history = []
        
        # Load checkpoint if resuming
        if resume_from:
            self.start_epoch = self._load_checkpoint(resume_from)
        
        # Initialize model
        self.model = YOLO(self.config['model']['name'])
        
        self.logger.info(f"Loaded YOLOv8n model from pretrained weights")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set default paths if not specified
        if 'checkpoint' not in config:
            config['checkpoint'] = {}
        if 'save_path' not in config['checkpoint']:
            config['checkpoint']['save_path'] = 'models/checkpoints'
        
        if 'logging' not in config:
            config['logging'] = {}
        if 'log_dir' not in config['logging']:
            config['logging']['log_dir'] = 'logs'
        if 'log_file' not in config['logging']:
            config['logging']['log_file'] = 'logs/train_yolov8.log'
        if 'metrics_file' not in config['logging']:
            config['logging']['metrics_file'] = 'logs/yolov8_metrics.json'
        
        return config
    
    def _setup_logging(self):
        """Setup logger to file and console."""
        log_file = self.config['logging']['log_file']
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger('YOLOv8Trainer')
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler(log_file, mode='a')
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def _load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load checkpoint and return starting epoch.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
        
        Returns:
            int: Epoch to resume from
        """
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return 0
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        epoch = checkpoint.get('epoch', 0)
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        self.metrics_history = checkpoint.get('metrics_history', [])
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch})")
        return epoch
    
    def _save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """
        Save training checkpoint.
        
        Args:
            epoch (int): Current epoch
            metrics (Dict): Metrics dictionary
            is_best (bool): Whether this is the best model so far
        """
        checkpoint_path = Path(self.config['checkpoint']['save_path'])
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch + 1,  # Save next epoch to resume from
            'best_metric': self.best_metric,
            'metrics_history': self.metrics_history,
            'config': self.config,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Save model weights
        model_path = checkpoint_path / f'yolov8n_epoch_{epoch:03d}.pt'
        self.model.save(str(model_path))
        self.logger.info(f"Saved checkpoint: {model_path}")
        
        # Save best model
        if is_best:
            best_path = checkpoint_path / 'yolov8n_best.pt'
            self.model.save(str(best_path))
            self.logger.info(f"Saved best model: {best_path}")
    
    def _save_metrics(self):
        """Save metrics history to JSON file."""
        metrics_path = self.config['logging']['metrics_file']
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def train(self, num_epochs: Optional[int] = None, batch_size: Optional[int] = None):
        """
        Train YOLOv8n on CAMEL dataset.
        
        Args:
            num_epochs (int): Number of epochs to train. If None, uses config value.
            batch_size (int): Batch size. If None, uses config value.
        """
        # Use provided args or fall back to config
        num_epochs = num_epochs or self.config['training']['epochs']
        batch_size = batch_size or self.config['training']['batch_size']
        
        # Load dataloaders
        dataset_path = self.config['data']['dataset_path']
        self.logger.info(f"Loading dataset from {dataset_path}")
        
        try:
            train_loader, val_loader = create_dataloaders(
                data_dir=dataset_path,
                batch_size=batch_size,
                num_workers=self.config['data']['num_workers'],
                augment=True,
                box_format='yolo',
            )
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise
        
        self.logger.info(f"Training set: {len(train_loader.dataset)} images")
        self.logger.info(f"Validation set: {len(val_loader.dataset)} images")
        self.logger.info(f"Starting training from epoch {self.start_epoch}")
        self.logger.info(f"Config: {json.dumps(self.config, indent=2)}")
        
        # Train with Ultralytics YOLO (uses native training pipeline)
        # This handles the training loop, validation, and loss calculation
        try:
            # Construct path to dataset YAML file
            dataset_yaml = Path(self.config['data']['dataset_path']) / 'camel.yaml'
            if not dataset_yaml.exists():
                # Try alternative name
                dataset_yaml = Path(self.config['data']['dataset_path']) / 'dataset.yaml'
            
            if not dataset_yaml.exists():
                raise FileNotFoundError(
                    f"Dataset YAML not found at {dataset_yaml}. "
                    f"Checked: data/camel/camel.yaml and data/camel/dataset.yaml"
                )
            
            self.logger.info(f"Using dataset config: {dataset_yaml}")
            
            results = self.model.train(
                data=str(dataset_yaml),  # YOLO expects path to dataset.yaml or camel.yaml
                epochs=num_epochs,
                imgsz=self.config['model']['img_size'],
                batch=batch_size,
                device=self.device.index or 0,
                patience=self.config['training']['patience'],
                save=True,
                save_period=self.config['checkpoint']['save_interval'],
                project='runs/yolov8',
                name='train',
                resume=bool(self.start_epoch > 0),
                pretrained=True,
                optimizer=self.config['training']['optimizer'],
                lr0=self.config['training']['learning_rate'],
                momentum=self.config['training']['momentum'],
                weight_decay=self.config['training']['weight_decay'],
                warmup_epochs=self.config['training']['warmup_epochs'],
                close_mosaic=15,
                mosaic=self.config['augmentation']['mosaic'],
                workers=3,
                verbose=True,
            )
            
            self.logger.info("Training completed successfully!")
            
            # Export final model
            if self.config['export']['onnx']:
                self._export_onnx()
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def _export_onnx(self):
        """Export trained model to ONNX format."""
        try:
            onnx_path = self.config['export']['onnx_path']
            Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
            
            self.model.export(
                format='onnx',
                imgsz=self.config['model']['img_size'],
                opset=self.config['export']['onnx_opset'],
            )
            
            self.logger.info(f"Exported model to ONNX: {onnx_path}")
        except Exception as e:
            self.logger.error(f"ONNX export failed: {e}")


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description='Train YOLOv8n on CAMEL infrared dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python scripts/train_yolov8.py --config configs/yolov8_config.yaml
  python scripts/train_yolov8.py --config configs/yolov8_config.yaml --resume models/checkpoints/yolov8n_epoch_50.pt
  python scripts/train_yolov8.py --config configs/yolov8_config.yaml --epochs 50 --batch-size 16
        ''',
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/yolov8_config.yaml',
        help='Path to config YAML file (default: configs/yolov8_config.yaml)',
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from (optional)',
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
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = YOLOv8Trainer(config_path=args.config, resume_from=args.resume)
    
    # Start training
    trainer.train(num_epochs=args.epochs, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
