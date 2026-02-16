"""
Faster R-CNN Training Script for CAMEL Infrared Dataset

Trains Faster R-CNN with MobileNetV3 backbone for object detection on infrared images.
Includes checkpoint management, logging, and validation.

Training runs are saved in incremental directories (train, train2, train3, etc.)
similar to Ultralytics YOLO, with all logs, checkpoints, and metrics in each run directory.

Usage:
    # Train from scratch
    python src/python/models/train_faster_rcnn.py
    
    # Train with custom epochs/batch size
    python src/python/models/train_faster_rcnn.py --epochs 50 --batch-size 16
    
    # Resume from last checkpoint (auto-detects)
    python src/python/models/train_faster_rcnn.py --resume
    
    # Resume from specific checkpoint
    python src/python/models/train_faster_rcnn.py --checkpoint models/checkpoints/fasterrcnn/train/faster_rcnn_epoch_30.pt
    
    # Custom run name
    python src/python/models/train_faster_rcnn.py --name experiment1

Training outputs saved to: models/checkpoints/fasterrcnn/{name}/
    - train.log - Training log file
    - metrics.json - Training metrics history
    - results.txt - Final results summary
    - config_used.yaml - Configuration used for this run
    - faster_rcnn_last.pt - Last training epoch
    - faster_rcnn_best.pt - Best validation loss
    - faster_rcnn_epoch_N.pt - Periodic checkpoints
"""

import argparse
import json
import logging
import os
import sys
import time
import yaml
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, default_collate

import torchvision.models as models
from torchvision.models.detection import FasterRCNN, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from tqdm import tqdm

# Add src/python directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.dataset import CAMELDataset
from dataset.faster_rcnn_augmentations import FasterRCNNAugmentations, InferenceAugmentations

class ModelEMA:
    """
    Exponential Moving Average for model weights.
    Maintains a shadow copy of model weights that are updated with:
        ema_weight = tau * ema_weight + (1 - tau) * current_weight
    
    Args:
        model: PyTorch model
        tau: Decay rate (default: 0.999)
    """
    def __init__(self, model: nn.Module, tau: float = 0.999):
        self.model = model
        self.tau = tau
        
        # Create shadow copy using state_dict (much faster than cloning params)
        self.shadow = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = param.detach().clone()
    
    @torch.no_grad()
    def update(self):
        """Update EMA weights after each training step."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                # Efficient in-place operation: ema = tau * ema + (1-tau) * param
                self.shadow[name].mul_(self.tau).add_(param.data, alpha=1.0 - self.tau)
    
    @torch.no_grad()
    def apply_shadow(self):
        """Replace model weights with EMA weights (for validation/inference)."""
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    @torch.no_grad()
    def restore(self):
        """Restore original model weights (after validation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
    
    def state_dict(self):
        """Return EMA state for checkpointing."""
        return {
            'shadow': self.shadow,
            'tau': self.tau,
        }
    
    def load_state_dict(self, state_dict):
        """Load EMA state from checkpoint."""
        self.shadow = state_dict['shadow']
        self.tau = state_dict.get('tau', self.tau)


# Setup logging
def setup_logging(log_file: str):
    """Configure logging to file and console."""
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str = 'configs/faster_rcnn_config.yaml') -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_faster_rcnn_model(num_classes: int, config: dict) -> FasterRCNN:
    """
    Build Faster R-CNN with MobileNetV3 Large backbone.
    
    Args:
        num_classes (int): Number of classes (background + object classes)
        config (dict): Model configuration
    
    Returns:
        FasterRCNN model
    """
    # When using custom num_classes, we need to set weights=None
    # then manually load backbone weights if desired
    model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=None,
        num_classes=num_classes,
        min_size=config['model'].get('min_size', 336),
        max_size=config['model'].get('max_size', 336),
        trainable_backbone_layers=config['model'].get('trainable_layers', 6),  # Fine-tune all layers of the backbone
    )
    # This initializes the backbone with ImageNet weights via the model's design
    return model


def collate_fn(batch):
    """
    Custom collate function for DataLoader to handle variable-sized bboxes.
    class label ids are incremented by 1 to account for background class (0) in Faster R-CNN.
    Assuems input batch is a list of (image, target) tuples where:
    - image: torch.Tensor of shape (C, H, W)
    - target: dict with keys 'boxes' (Tensor of shape [num_boxes, 4]) and 'class_ids' (Tensor of shape [num_boxes])

    Args:
        batch: List of (image, target) tuples
    
    Returns:
        Tuple of (images, targets)
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        
        # Safeguard: Handle empty bounding boxes (can occur after aggressive augmentation)
        boxes = target['boxes'].float()
        class_ids = target['class_ids'].long()
        
        # Ensure proper shape even for empty tensors
        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            class_ids = torch.zeros((0,), dtype=torch.long)
        
        # Convert target to Faster R-CNN format
        target_dict = {
            'boxes': boxes,
            'labels': class_ids + 1,  # Add 1 for background class
        }
        targets.append(target_dict)
    
    return images, targets


class CAMELFasterRCNNDataset(CAMELDataset):
    """Extended CAMEL dataset with augmentations for Faster R-CNN."""
    
    def __init__(self, root_dir: str, split: str = "train", augment: bool = True, config: dict = None):
        """Initialize with Faster R-CNN specific settings."""
        super().__init__(
            root_dir=root_dir,
            split=split,
            augment=augment,
            normalize=False,  # Faster R-CNN augmentations handle normalization
            box_format='yolo',  # Use YOLO format (normalized) for albumentations
        )
        
        self.config = config or {}
        self.split = split
        
        # Setup augmentations
        if split == "train" and augment:
            self.augmentation = FasterRCNNAugmentations(
                config=self.config.get('augmentation', {}),
                img_size=(self.image_height, self.image_width)
            )
        else:
            self.augmentation = InferenceAugmentations(
                img_size=(self.image_height, self.image_width)
            )
    
    def __getitem__(self, idx: int):
        """Load image and apply augmentations."""
        img_path = self.image_files[idx]
        
        # Load image
        image = self._load_image(img_path)
        
        # Load labels
        targets = self._load_labels(img_path)
        
        # Convert boxes to list for augmentation pipeline
        boxes = targets['boxes'].tolist() if len(targets['boxes']) > 0 else []
        class_ids = targets['class_ids'].tolist() if len(targets['class_ids']) > 0 else []
        
        # Validate and clean boxes before augmentation (filter out invalid boxes)
        # YOLO format: [x_center, y_center, width, height] normalized
        valid_boxes = []
        valid_class_ids = []
        for box, cls_id in zip(boxes, class_ids):
            x_center, y_center, width, height = box
            
            # Check for positive dimensions
            if width <= 0.0 or height <= 0.0:
                continue
            
            # Calculate corners
            x_min = x_center - width / 2
            x_max = x_center + width / 2
            y_min = y_center - height / 2
            y_max = y_center + height / 2
            
            # Clip to valid range [0, 1]
            x_min = max(0.0, min(1.0, x_min))
            y_min = max(0.0, min(1.0, y_min))
            x_max = max(0.0, min(1.0, x_max))
            y_max = max(0.0, min(1.0, y_max))
            
            # Recompute center and dimensions after clipping
            width_clipped = x_max - x_min
            height_clipped = y_max - y_min
            
            # Check if box is still valid after clipping (minimum size threshold)
            if width_clipped >= 0.001 and height_clipped >= 0.001:
                x_center_clipped = (x_min + x_max) / 2
                y_center_clipped = (y_min + y_max) / 2
                valid_boxes.append([x_center_clipped, y_center_clipped, width_clipped, height_clipped])
                valid_class_ids.append(cls_id)
        
        boxes = valid_boxes
        class_ids = valid_class_ids
        
        # Cache original image for fallback
        image_original = image.copy()
        
        # Apply augmentations with error handling
        try:
            image, boxes, class_ids = self.augmentation(image, boxes, class_ids)
        except (ValueError, AssertionError) as e:
            # If augmentation fails, use cached original image without augmentation
            print(f"Warning: Augmentation failed for {img_path.name}: {e}")
            print(f"         Using original image without augmentation. Valid boxes: {len(boxes)}")
            
            # Convert cached image to tensor format manually
            image = image_original
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)
            if image.dtype != np.float32:
                if image.max() > 1.0:
                    image = image.astype(np.float32) / 255.0
                else:
                    image = image.astype(np.float32)
            image = np.clip(image, 0.0, 1.0)
            # Convert to tensor: (H, W, 1) -> (1, H, W)
            image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Safeguard: Validate augmentation output (boxes can be filtered out by augmentation)
        if not boxes or len(boxes) == 0:
            # No objects remain after augmentation - return empty tensors
            targets['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            targets['class_ids'] = torch.zeros((0,), dtype=torch.long)
        else:
            # Convert YOLO format (normalized) to corner format (pixels) for Faster R-CNN
            corner_boxes = []
            final_class_ids = []
            for box, cls_id in zip(boxes, class_ids):
                x_center, y_center, width, height = box
                
                # Validation already done, but check for edge cases
                if width <= 0 or height <= 0:
                    continue
                
                x1 = (x_center - width / 2) * self.image_width
                y1 = (y_center - height / 2) * self.image_height
                x2 = (x_center + width / 2) * self.image_width
                y2 = (y_center + height / 2) * self.image_height
                
                # Clip to image boundaries
                x1 = max(0, min(self.image_width, x1))
                y1 = max(0, min(self.image_height, y1))
                x2 = max(0, min(self.image_width, x2))
                y2 = max(0, min(self.image_height, y2))
                
                # Final validation: ensure at least 1 pixel in each dimension
                if x2 > x1 + 1.0 and y2 > y1 + 1.0:
                    corner_boxes.append([x1, y1, x2, y2])
                    final_class_ids.append(cls_id)
            
            # Convert to tensors (lengths must match by construction)
            if corner_boxes:
                targets['boxes'] = torch.tensor(corner_boxes, dtype=torch.float32)
                targets['class_ids'] = torch.tensor(final_class_ids, dtype=torch.long)
            else:
                targets['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                targets['class_ids'] = torch.zeros((0,), dtype=torch.long)
        
        return image, targets


def train_epoch(model, data_loader, optimizer, device, config: dict, epoch: int, logger, ema=None) -> float:
    """
    Train for one epoch.
    
    Args:
        model: Faster R-CNN model
        data_loader: Training DataLoader
        optimizer: Optimizer
        device: Device (cuda/cpu)
        config: Configuration dict
        epoch: Epoch number
        logger: Logger instance
        ema: ModelEMA instance (optional)
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1} - Training")
    
    for batch_idx, (images, targets) in enumerate(pbar):
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Update EMA
        if ema is not None:
            ema.update()
        
        # Accumulate loss
        total_loss += losses.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': total_loss / num_batches})
        
        # Log metrics using tqdm.write() to avoid duplicating progress bar
        if (batch_idx + 1) % config['logging'].get('log_interval', 10) == 0:
            avg_loss = total_loss / num_batches
            log_msg = (
                f"Epoch {epoch+1} - Batch {batch_idx+1}/{len(data_loader)} - "
                f"Loss: {losses.item():.4f} - Avg Loss: {avg_loss:.4f}"
            )
            # Print above progress bar
            tqdm.write(log_msg)
            # Log to file (without console output to avoid duplication)
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.emit(logger.makeRecord(
                        logger.name, logging.INFO, __file__, 0,
                        log_msg, (), None
                    ))
    
    return total_loss / num_batches


@torch.no_grad()
def validate(model, data_loader, device, config: dict, epoch: int, logger, ema=None) -> float:
    """
    Validate the model.
    
    Args:
        model: Faster R-CNN model
        data_loader: Validation DataLoader
        device: Device (cuda/cpu)
        config: Configuration dict
        epoch: Epoch number
        logger: Logger instance
        ema: ModelEMA instance (optional, uses EMA weights if provided)
    
    Returns:
        Average validation loss
    """
    # Apply EMA weights for validation if available
    if ema is not None:
        ema.apply_shadow()
    
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1} - Validation")
    
    for images, targets in pbar:
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Note: Faster R-CNN only returns loss_dict in train mode
        # In eval mode, it returns predictions instead of losses
        # We temporarily switch to train mode to compute validation loss
        model.train()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        model.eval()
        
        # Accumulate loss
        total_loss += losses.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'val_loss': total_loss / num_batches})
    
    avg_val_loss = total_loss / num_batches
    log_msg = f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}"
    # Print above progress bar
    tqdm.write(log_msg)
    # Log to file (without console output to avoid duplication)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.emit(logger.makeRecord(
                logger.name, logging.INFO, __file__, 0,
                log_msg, (), None
            ))
    
    # Restore original weights after validation
    if ema is not None:
        ema.restore()
    
    return avg_val_loss


def find_last_checkpoint(checkpoint_dir: Path) -> Path:
    """
    Find the most recent checkpoint in the directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
    
    Returns:
        Path to the last checkpoint, or None if no checkpoints found
    """
    # First try to find 'faster_rcnn_last.pt'
    last_checkpoint = checkpoint_dir / "faster_rcnn_last.pt"
    if last_checkpoint.exists():
        return last_checkpoint
    
    # Otherwise, find the checkpoint with highest epoch number
    epoch_checkpoints = list(checkpoint_dir.glob("faster_rcnn_epoch_*.pt"))
    if epoch_checkpoints:
        # Extract epoch numbers and find max
        def get_epoch_num(path: Path) -> int:
            try:
                return int(path.stem.split('_')[-1])
            except (ValueError, IndexError):
                return -1
        
        return max(epoch_checkpoints, key=get_epoch_num)
    
    return None


def get_next_run_dir(base_dir: Path, name: str = 'train') -> Path:
    """
    Get the next available run directory (train, train2, train3, etc.)
    Similar to Ultralytics YOLO behavior.
    
    Args:
        base_dir: Base checkpoint directory (e.g., models/checkpoints/fasterrcnn)
        name: Base name for run directory (default: 'train')
    
    Returns:
        Path to the next available run directory
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if base name is available
    run_dir = base_dir / name
    if not run_dir.exists():
        return run_dir
    
    # Find next available numbered directory
    i = 2
    while True:
        run_dir = base_dir / f"{name}{i}"
        if not run_dir.exists():
            return run_dir
        i += 1


def train_faster_rcnn(
    epochs: int = None,
    batch_size: int = None,
    resume: bool = False,
    checkpoint_path: str = None,
    config_path: str = 'configs/faster_rcnn_config.yaml',
    name: str = 'train',
):
    """
    Train Faster R-CNN on CAMEL infrared dataset.
    
    Args:
        epochs (int): Number of epochs. If None, uses config value.
        batch_size (int): Batch size. If None, uses config value.
        resume (bool): Resume from last training. Default: False
        checkpoint_path (str): Path to checkpoint to resume from
        config_path (str): Path to config YAML file
        name (str): Name for this training run (default: 'train')
    """
    # Load configuration
    config = load_config(config_path)
    
    # Override config with CLI args if provided
    epochs = epochs or config['training']['epochs']
    batch_size = batch_size or config['training']['batch_size']
    
    # Setup run directory (train, train2, train3, etc.)
    base_checkpoint_dir = Path("models/checkpoints/fasterrcnn")
    
    # If resuming, use existing checkpoint directory
    if resume or checkpoint_path:
        if checkpoint_path:
            run_dir = Path(checkpoint_path).parent
        else:
            # Find most recent run directory
            existing_runs = sorted(base_checkpoint_dir.glob(f"{name}*"))
            if existing_runs:
                run_dir = existing_runs[-1]
            else:
                run_dir = get_next_run_dir(base_checkpoint_dir, name)
    else:
        # Create new run directory
        run_dir = get_next_run_dir(base_checkpoint_dir, name)
    
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to run directory
    log_file = run_dir / "train.log"
    logger = setup_logging(str(log_file))
    logger.info("=" * 80)
    logger.info("Starting Faster R-CNN Training")
    logger.info("=" * 80)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Config file: {config_path}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    
    # Device
    device = torch.device(f"cuda:{config['model']['device']}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Create datasets
    logger.info("Loading datasets...")
    data_dir = config['data']['dataset_path']
    
    train_dataset = CAMELFasterRCNNDataset(
        root_dir=data_dir,
        split='train',
        augment=True,
        config=config
    )
    
    val_dataset = CAMELFasterRCNNDataset(
        root_dir=data_dir,
        split='val',
        augment=False,
        config=config
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    # Only use pin_memory if GPU is available
    use_pin_memory = config['data']['pin_memory'] and torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config['data']['num_workers'],
        shuffle=True,
        pin_memory=use_pin_memory,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=config['data']['num_workers'],
        shuffle=False,
        pin_memory=use_pin_memory,
        collate_fn=collate_fn,
    )
    
    # Build model
    logger.info("Building model...")
    num_classes = config['data']['num_classes']
    model = build_faster_rcnn_model(num_classes, config)
    model.to(device)
    logger.info(f"Model: Faster R-CNN with MobileNetV3 Large ({num_classes} classes)")
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=config['training']['learning_rate'],
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=30,
        gamma=0.1
    )
    
    # Setup EMA (Exponential Moving Average) if enabled
    ema = None
    if config['training'].get('ema', False):
        ema_tau = config['training'].get('ema_tau', 0.999)
        ema = ModelEMA(model, tau=ema_tau)
        logger.info(f"EMA enabled with tau={ema_tau}")
    
    # Checkpoint directory is the run directory
    checkpoint_dir = run_dir
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    
    # Save config to run directory for reproducibility
    config_save_path = run_dir / "config_used.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Config saved to: {config_save_path}")
    
    start_epoch = 0
    best_val_loss = float('inf')
    metrics_history = {'train_loss': [], 'val_loss': []}
    
    # Resume from checkpoint if specified
    if resume or checkpoint_path:
        # Auto-find last checkpoint if resume=True but no path specified
        if resume and not checkpoint_path:
            last_checkpoint = find_last_checkpoint(checkpoint_dir)
            if last_checkpoint:
                checkpoint_path = str(last_checkpoint)
                logger.info(f"Auto-detected last checkpoint: {checkpoint_path}")
            else:
                logger.warning("Resume requested but no checkpoint found. Starting from scratch.")
        
        if checkpoint_path and Path(checkpoint_path).exists():
            logger.info("=" * 80)
            logger.info("RESUMING FROM CHECKPOINT")
            logger.info("=" * 80)
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state if available
            if 'lr_scheduler_state_dict' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                logger.info("Loaded learning rate scheduler state")
            
            # Load EMA state if available
            if ema is not None and 'ema_state_dict' in checkpoint:
                ema.load_state_dict(checkpoint['ema_state_dict'])
                logger.info("Loaded EMA state")
            
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            metrics_history = checkpoint.get('metrics_history', metrics_history)
            
            logger.info(f"Resuming from epoch {start_epoch}")
            logger.info(f"Best validation loss so far: {best_val_loss:.4f}")
            logger.info(f"Training history: {len(metrics_history['train_loss'])} epochs")
            logger.info("=" * 80)
        elif checkpoint_path:
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            logger.info("Starting training from scratch...")
    
    # Training loop
    logger.info("Starting training...")
    patience_counter = 0
    max_patience = config['training']['patience']
    last_epoch = start_epoch - 1  # Track last completed epoch
    
    for epoch in range(start_epoch, epochs):
        last_epoch = epoch  # Update after each epoch starts
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, config, epoch, logger, ema=ema)
        metrics_history['train_loss'].append(train_loss)
        
        # Validate
        if (epoch + 1) % config['validation']['val_interval'] == 0:
            val_loss = validate(model, val_loader, device, config, epoch, logger, ema=ema)
            metrics_history['val_loss'].append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                if config['checkpoint'].get('save_best', True):
                    best_checkpoint_path = checkpoint_dir / f"faster_rcnn_best.pt"
                    checkpoint_data = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                        'best_val_loss': best_val_loss,
                        'metrics_history': metrics_history,
                        'config': config,
                    }
                    if ema is not None:
                        checkpoint_data['ema_state_dict'] = ema.state_dict()
                    torch.save(checkpoint_data, best_checkpoint_path)
                    logger.info(f"Best model saved: {best_checkpoint_path} (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
            
            if patience_counter >= max_patience:
                logger.info(f"Early stopping triggered after {max_patience} epochs without improvement")
                break
        
        # Save checkpoint at intervals
        if (epoch + 1) % config['checkpoint']['save_interval'] == 0:
            interval_checkpoint_path = checkpoint_dir / f"faster_rcnn_epoch_{epoch+1}.pt"
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'metrics_history': metrics_history,
                'config': config,
            }
            if ema is not None:
                checkpoint_data['ema_state_dict'] = ema.state_dict()
            torch.save(checkpoint_data, interval_checkpoint_path)
            logger.info(f"Checkpoint saved: {interval_checkpoint_path}")
        
        # Learning rate scheduling
        lr_scheduler.step()
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Save last model
    if config['checkpoint'].get('save_last', True):
        last_checkpoint_path = checkpoint_dir / "faster_rcnn_last.pt"
        checkpoint_data = {
            'epoch': last_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'metrics_history': metrics_history,
            'config': config,
        }
        if ema is not None:
            checkpoint_data['ema_state_dict'] = ema.state_dict()
        torch.save(checkpoint_data, last_checkpoint_path)
        logger.info(f"Last model saved: {last_checkpoint_path}")
    
    # Save metrics to run directory
    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics_history, f, indent=2)
    logger.info(f"Metrics saved to: {metrics_file}")
    
    # Save final results summary
    results_file = run_dir / "results.txt"
    with open(results_file, 'w') as f:
        f.write("Faster R-CNN Training Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Epochs: {len(metrics_history['train_loss'])}\n")
        f.write(f"Best Validation Loss: {best_val_loss:.4f}\n")
        f.write(f"Final Training Loss: {metrics_history['train_loss'][-1]:.4f}\n")
        if metrics_history['val_loss']:
            f.write(f"Final Validation Loss: {metrics_history['val_loss'][-1]:.4f}\n")
    logger.info(f"Results summary saved to: {results_file}")
    
    logger.info("Training completed!")
    logger.info("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Faster R-CNN on CAMEL dataset')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--config', type=str, default='configs/faster_rcnn_config.yaml', help='Config file path')
    parser.add_argument('--name', type=str, default='train', help='Name for this training run (default: train)')
    
    args = parser.parse_args()
    
    train_faster_rcnn(
        epochs=args.epochs,
        batch_size=args.batch_size,
        resume=args.resume,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        name=args.name,
    )
