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
    python src/python/models/train_faster_rcnn.py --checkpoint models/checkpoints/fasterrcnn/train/weights/last.pt

    # Custom run name
    python src/python/models/train_faster_rcnn.py --name experiment1

Training outputs saved to: models/checkpoints/fasterrcnn/{name}/
    - args.yaml - Configuration used for this run
    - train.log - Training log file
    - results.csv - Training metrics per epoch (YOLO-style format)
    - results.png - Training curves visualization
    - metrics.json - Training metrics history (detailed)
    - results.txt - Final results summary
    - weights/
        - best.pt - Best validation loss checkpoint
        - best.onnx - Exported ONNX model (from best.pt)
        - last.pt - Last training epoch
        - epochX.pt - Periodic checkpoints
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import argparse
import csv
import json
import logging
import os
import sys
import time
import yaml
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, default_collate

import torchvision.models as models
from torchvision.models.detection import FasterRCNN, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou

from tqdm import tqdm

# Add src/python directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.dataset import CAMELDataset
from dataset.faster_rcnn_augmentations import FasterRCNNAugmentations, InferenceAugmentations


# ==============================================================================
# CONFIGURATION AND SETUP
# ==============================================================================

def load_config(config_path: str = 'configs/faster_rcnn_config.yaml') -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(log_file: str):
    """
    Configure logging to file and console.

    Args:
        log_file: Path to log file

    Returns:
        Logger instance
    """
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


# ==============================================================================
# MODEL COMPONENTS
# ==============================================================================

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

def build_faster_rcnn_model(num_classes: int, config: dict) -> FasterRCNN:
    """
    Build Faster R-CNN with MobileNetV3 Large backbone with pretrained weights.

    Args:
        num_classes: Number of classes (background + object classes)
        config: Model configuration

    Returns:
        FasterRCNN model
    """
    # Use pretrained weights on COCO (better initialization than random weights)
    use_pretrained = config['model'].get('pretrained', True)

    if use_pretrained:
        # Load model with COCO pretrained weights (91 classes)
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights=weights,
            min_size=config['model'].get('min_size', 336),
            max_size=config['model'].get('max_size', 336),
            trainable_backbone_layers=config['model'].get('trainable_layers', 6),
        )

        # Replace the classification head with custom number of classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        print(f"✓ Loaded pretrained COCO weights for backbone and RPN")
        print(f"✓ Replaced classification head for {num_classes} classes")
    else:
        # Train from scratch (not recommended)
        model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights=None,
            num_classes=num_classes,
            min_size=config['model'].get('min_size', 336),
            max_size=config['model'].get('max_size', 336),
            trainable_backbone_layers=config['model'].get('trainable_layers', 6),
        )
        print(f"⚠ Training from scratch (no pretrained weights)")

    return model

# ==============================================================================
# DATASET AND DATA LOADING
# ==============================================================================

def collate_fn(batch):
    """
    Custom collate function for DataLoader to handle variable-sized bboxes.
    Class label IDs are incremented by 1 to account for background class (0) in Faster R-CNN.

    Args:
        batch: List of (image, target) tuples where:
            - image: torch.Tensor of shape (C, H, W)
            - target: dict with keys 'boxes' and 'class_ids'

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

    def __init__(self, root_dir: str, split: str = "train", augment: bool = True, config: dict = None, cache_images: bool = False):
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
        self.cache_images = cache_images
        self._image_cache = {} if cache_images else None

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

        # Cache images if requested
        if self.cache_images:
            print(f"Caching {len(self.image_files)} images for {split} split...")
            for idx, img_path in enumerate(tqdm(self.image_files, desc=f"Caching {split}")):
                self._image_cache[idx] = self._load_image(img_path)

    def __getitem__(self, idx: int):
        """Load image and apply augmentations."""
        img_path = self.image_files[idx]

        # Load image (from cache or disk)
        if self.cache_images:
            image = self._image_cache[idx].copy()  # Copy cached image for augmentation
        else:
            image = self._load_image(img_path)

        # Load labels
        targets = self._load_labels(img_path)

        # Convert boxes to list for augmentation pipeline (albumentations expects lists)
        boxes = targets['boxes'].tolist() if len(targets['boxes']) > 0 else []
        class_ids = targets['class_ids'].tolist() if len(targets['class_ids']) > 0 else []

        # Validate and clean boxes before augmentation (vectorized)
        # This step ensures that we don't pass invalid boxes to the augmentation pipeline, which can cause errors.
        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            class_ids_tensor = torch.tensor(class_ids, dtype=torch.long)

            # Extract box components
            x_center = boxes_tensor[:, 0]
            y_center = boxes_tensor[:, 1]
            width = boxes_tensor[:, 2]
            height = boxes_tensor[:, 3]

            # Filter out boxes with non-positive dimensions
            valid_dims = (width > 0.0) & (height > 0.0)

            # Calculate corners and clip to valid range [0, 1]
            x_min = torch.clamp(x_center - width / 2, 0.0, 1.0)
            y_min = torch.clamp(y_center - height / 2, 0.0, 1.0)
            x_max = torch.clamp(x_center + width / 2, 0.0, 1.0)
            y_max = torch.clamp(y_center + height / 2, 0.0, 1.0)

            # Recompute dimensions after clipping
            width_clipped = x_max - x_min
            height_clipped = y_max - y_min

            # Check if box is still valid after clipping (minimum size threshold)
            valid_size = (width_clipped >= 0.001) & (height_clipped >= 0.001)

            # Combined validity mask
            valid_mask = valid_dims & valid_size

            if valid_mask.any():
                # Recompute centers after clipping
                x_center_clipped = (x_min + x_max) / 2
                y_center_clipped = (y_min + y_max) / 2

                # Stack valid boxes and convert to list for albumentations
                valid_boxes_tensor = torch.stack([
                    x_center_clipped[valid_mask],
                    y_center_clipped[valid_mask],
                    width_clipped[valid_mask],
                    height_clipped[valid_mask]
                ], dim=1)

                boxes = valid_boxes_tensor.tolist()
                class_ids = class_ids_tensor[valid_mask].tolist()
            else:
                boxes = []
                class_ids = []

        # Apply augmentations with error handling (only for training, validation uses simple transforms)
        if self.split == "train":
            try:
                image, boxes, class_ids = self.augmentation(image, boxes, class_ids)
            except (ValueError, AssertionError) as e:
                # If augmentation fails, use original image with basic preprocessing
                print(f"Warning: Augmentation failed for {img_path.name}: {e}")
                print(f"         Using fallback preprocessing. Valid boxes: {len(boxes)}")

                # Basic preprocessing without augmentation
                if len(image.shape) == 2:
                    image = np.expand_dims(image, axis=-1)
                if image.dtype != np.float32:
                    if image.max() > 1.0:
                        image = image.astype(np.float32) / 255.0
                    else:
                        image = image.astype(np.float32)
                image = np.clip(image, 0.0, 1.0)
                image = torch.from_numpy(image).permute(2, 0, 1)
        else:
            # Validation: always use simple augmentation (no failures expected)
            image, boxes, class_ids = self.augmentation(image, boxes, class_ids)

        # Safeguard: Validate augmentation output
        if not boxes or len(boxes) == 0:
            # No objects remain after augmentation - return empty tensors
            targets['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            targets['class_ids'] = torch.zeros((0,), dtype=torch.long)
        else:
            # Convert YOLO format (normalized) to corner format (pixels) for Faster R-CNN (vectorized)

            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            class_ids_tensor = torch.tensor(class_ids, dtype=torch.long)

            # Extract box components
            x_center = boxes_tensor[:, 0]
            y_center = boxes_tensor[:, 1]
            width = boxes_tensor[:, 2]
            height = boxes_tensor[:, 3]

            # Filter out invalid dimensions
            valid_dims = (width > 0) & (height > 0)

            if valid_dims.any():
                # Convert to corner format (pixels)
                x1 = (x_center - width / 2) * self.image_width
                y1 = (y_center - height / 2) * self.image_height
                x2 = (x_center + width / 2) * self.image_width
                y2 = (y_center + height / 2) * self.image_height

                # Clip to image boundaries
                x1 = torch.clamp(x1, 0, self.image_width)
                y1 = torch.clamp(y1, 0, self.image_height)
                x2 = torch.clamp(x2, 0, self.image_width)
                y2 = torch.clamp(y2, 0, self.image_height)

                # Final validation: ensure at least 1 pixel in each dimension
                valid_final = (x2 > x1 + 1.0) & (y2 > y1 + 1.0) & valid_dims

                if valid_final.any():
                    # Stack valid boxes
                    corner_boxes = torch.stack([x1[valid_final], y1[valid_final], x2[valid_final], y2[valid_final]], dim=1)
                    targets['boxes'] = corner_boxes
                    targets['class_ids'] = class_ids_tensor[valid_final]
                else:
                    targets['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                    targets['class_ids'] = torch.zeros((0,), dtype=torch.long)
            else:
                targets['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                targets['class_ids'] = torch.zeros((0,), dtype=torch.long)

        return image, targets

# ==============================================================================
# TRAINING CORE
# ==============================================================================

def train_epoch(model, data_loader, optimizer, device, config: dict, epoch: int, logger, ema=None, scaler=None) -> dict:
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
        scaler: GradScaler for mixed precision (optional)

    Returns:
        Dict with average losses for the epoch
    """
    model.train()

    total_loss = 0.0
    total_loss_classifier = 0.0
    total_loss_box_reg = 0.0
    total_loss_objectness = 0.0
    total_loss_rpn_box_reg = 0.0
    num_batches = 0

    # Get gradient settings from config
    max_grad_norm = config['training'].get('max_grad_norm', 10.0)
    log_interval = config['logging'].get('log_interval', 50)  # Reduced default logging frequency

    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1} - Training")

    for batch_idx, (images, targets) in enumerate(pbar):
        # Move to device (use non_blocking for async transfer)
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)

        # Check for NaN in losses before backward pass
        has_nan = any(torch.isnan(v) or torch.isinf(v) for v in loss_dict.values())

        if has_nan:
            if (batch_idx + 1) % log_interval == 0:
                logger.warning(f"Skipping batch {batch_idx+1} due to NaN/Inf in losses")
            continue

        losses = sum(loss for loss in loss_dict.values())

        # Check total loss
        if torch.isnan(losses) or torch.isinf(losses):
            if (batch_idx + 1) % log_interval == 0:
                logger.warning(f"Skipping batch {batch_idx+1} due to NaN/Inf in total loss")
            continue

        # Backward pass
        optimizer.zero_grad()
        losses.backward()

        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        # Update EMA
        if ema is not None:
            ema.update()

        # Accumulate losses (use .item() sparingly to reduce CPU-GPU sync)
        total_loss += losses.item()
        total_loss_classifier += loss_dict.get('loss_classifier', torch.tensor(0.0)).item()
        total_loss_box_reg += loss_dict.get('loss_box_reg', torch.tensor(0.0)).item()
        total_loss_objectness += loss_dict.get('loss_objectness', torch.tensor(0.0)).item()
        total_loss_rpn_box_reg += loss_dict.get('loss_rpn_box_reg', torch.tensor(0.0)).item()
        num_batches += 1

        # Update progress bar less frequently (every 10 batches)
        if batch_idx % 10 == 0:
            avg_loss = total_loss / num_batches
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'cls': f'{total_loss_classifier / num_batches:.4f}',
                'box': f'{total_loss_box_reg / num_batches:.4f}',
            })

        # Reduced logging frequency
        if (batch_idx + 1) % log_interval == 0:
            avg_cls = total_loss_classifier / num_batches
            avg_box = total_loss_box_reg / num_batches
            avg_obj = total_loss_objectness / num_batches
            avg_rpn = total_loss_rpn_box_reg / num_batches
            avg_loss = total_loss / num_batches

            log_msg = (
                f"Epoch {epoch+1} - Batch {batch_idx+1}/{len(data_loader)} - "
                f"Loss: {losses.item():.4f} - Avg: {avg_loss:.4f} - "
                f"Cls: {avg_cls:.4f} - Box: {avg_box:.4f} - Obj: {avg_obj:.4f} - RPN: {avg_rpn:.4f}"
            )
            tqdm.write(log_msg)
            # Log to file only
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.emit(logger.makeRecord(
                        logger.name, logging.INFO, __file__, 0,
                        log_msg, (), None
                    ))

    # Return average losses
    if num_batches > 0:
        return {
            'total_loss': total_loss / num_batches,
            'loss_classifier': total_loss_classifier / num_batches,
            'loss_box_reg': total_loss_box_reg / num_batches,
            'loss_objectness': total_loss_objectness / num_batches,
            'loss_rpn_box_reg': total_loss_rpn_box_reg / num_batches,
        }
    else:
        return {
            'total_loss': 0.0,
            'loss_classifier': 0.0,
            'loss_box_reg': 0.0,
            'loss_objectness': 0.0,
            'loss_rpn_box_reg': 0.0,
        }


def validate(model, data_loader, device, config: dict, epoch: int, logger, ema=None) -> dict:
    """
    Validate the model with detection metrics.

    Args:
        model: Faster R-CNN model
        data_loader: Validation DataLoader
        device: Device (cuda/cpu)
        config: Configuration dict
        epoch: Epoch number
        logger: Logger instance
        ema: ModelEMA instance (optional, uses EMA weights if provided)

    Returns:
        Dict with validation losses and detection metrics
    
    Note:
        This function does NOT use @torch.no_grad() decorator because Faster R-CNN
        requires gradient tracking to be enabled (even without backprop) when computing
        losses in training mode. We use torch.no_grad() selectively for inference only.
    """
    # Apply EMA weights for validation if available
    if ema is not None:
        ema.apply_shadow()

    total_loss = 0.0
    total_loss_classifier = 0.0
    total_loss_box_reg = 0.0
    total_loss_objectness = 0.0
    total_loss_rpn_box_reg = 0.0
    num_batches = 0

    # For detection metrics
    all_predictions = []
    all_targets = []

    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1} - Validation")

    for images, targets in pbar:
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Skip batches where all images have no targets (can cause NaN in loss)
        valid_targets = [t for t in targets if len(t['boxes']) > 0]
        if len(valid_targets) == 0:
            logger.warning(f"Skipping batch with no valid targets at epoch {epoch+1}")
            continue

        # Compute validation loss (requires train mode for Faster R-CNN)
        model.train()
        with torch.no_grad():  # Use no_grad to prevent gradient tracking during validation
            try:
                loss_dict = model(images, targets)

                # Check for NaN in losses
                has_nan = False
                for loss_name, loss_value in loss_dict.items():
                    if torch.isnan(loss_value) or torch.isinf(loss_value):
                        logger.warning(f"NaN/Inf detected in validation {loss_name} at epoch {epoch+1}")
                        has_nan = True

                if has_nan:
                    logger.warning("Skipping validation batch due to NaN/Inf in losses")
                    model.eval()
                    continue

                losses = sum(loss for loss in loss_dict.values())

                # Check total loss
                if torch.isnan(losses) or torch.isinf(losses):
                    logger.warning(f"NaN/Inf detected in validation total loss at epoch {epoch+1}")
                    model.eval()
                    continue

                # Accumulate losses
                total_loss += losses.item()
                total_loss_classifier += loss_dict.get('loss_classifier', torch.tensor(0.0)).item()
                total_loss_box_reg += loss_dict.get('loss_box_reg', torch.tensor(0.0)).item()
                total_loss_objectness += loss_dict.get('loss_objectness', torch.tensor(0.0)).item()
                total_loss_rpn_box_reg += loss_dict.get('loss_rpn_box_reg', torch.tensor(0.0)).item()
                num_batches += 1

            except Exception as e:
                logger.error(f"Error during validation loss computation: {e}")
                model.eval()
                continue

        # Get predictions for metrics (requires eval mode)
        model.eval()
        with torch.no_grad():  # Use no_grad for inference to save memory
            try:
                predictions = model(images)

                # Store predictions and targets for metrics
                all_predictions.extend(predictions)
                all_targets.extend(targets)

            except Exception as e:
                logger.error(f"Error during validation inference: {e}")
                continue

        # Update progress bar
        if num_batches > 0:
            pbar.set_postfix({'val_loss': total_loss / num_batches})

    # Compute average losses
    if num_batches > 0:
        avg_val_loss = total_loss / num_batches
        avg_cls = total_loss_classifier / num_batches
        avg_box = total_loss_box_reg / num_batches
        avg_obj = total_loss_objectness / num_batches
        avg_rpn = total_loss_rpn_box_reg / num_batches
    else:
        logger.error("No valid batches during validation - all batches had NaN/Inf")
        avg_val_loss = float('nan')
        avg_cls = float('nan')
        avg_box = float('nan')
        avg_obj = float('nan')
        avg_rpn = float('nan')

    # Compute detection metrics
    metrics_result = {'precision': 0.0, 'recall': 0.0, 'mAP50': 0.0, 'mAP50_95': 0.0}
    if len(all_predictions) > 0 and len(all_targets) > 0:
        try:
            conf_threshold = config['validation'].get('conf_threshold', 0.25)
            iou_threshold = config['validation'].get('iou_threshold', 0.5)
            num_classes = config['data']['num_classes']

            metrics_result = compute_detection_metrics(
                all_predictions, all_targets,
                iou_threshold=iou_threshold,
                conf_threshold=conf_threshold,
                num_classes=num_classes
            )
        except Exception as e:
            logger.error(f"Error computing detection metrics: {e}")

    # Log results
    log_msg = (
        f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f} - "
        f"Cls: {avg_cls:.4f} - Box: {avg_box:.4f} - Obj: {avg_obj:.4f} - RPN: {avg_rpn:.4f} - "
        f"P: {metrics_result['precision']:.3f} - R: {metrics_result['recall']:.3f} - "
        f"mAP50: {metrics_result['mAP50']:.3f} - mAP50-95: {metrics_result['mAP50_95']:.3f}"
    )
    tqdm.write(log_msg)
    # Log to file only
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.emit(logger.makeRecord(
                logger.name, logging.INFO, __file__, 0,
                log_msg, (), None
            ))

    # Restore original weights after validation
    if ema is not None:
        ema.restore()

    return {
        'total_loss': avg_val_loss,
        'loss_classifier': avg_cls,
        'loss_box_reg': avg_box,
        'loss_objectness': avg_obj,
        'loss_rpn_box_reg': avg_rpn,
        'precision': metrics_result['precision'],
        'recall': metrics_result['recall'],
        'mAP50': metrics_result['mAP50'],
        'mAP50_95': metrics_result['mAP50_95'],
    }

# ==============================================================================
# METRICS COMPUTATION
# ==============================================================================

def compute_detection_metrics(predictions, targets, iou_threshold=0.5, conf_threshold=0.25, num_classes=5):
    """
    Compute detection metrics similar to YOLO: Precision, Recall, mAP50, mAP50-95.

    Args:
        predictions: List of dicts with keys 'boxes', 'labels', 'scores' for each image
        targets: List of dicts with keys 'boxes', 'labels' for each image
        iou_threshold: IoU threshold for mAP50 (default: 0.5)
        conf_threshold: Confidence threshold for filtering predictions
        num_classes: Number of classes (including background)

    Returns:
        Dict with metrics: precision, recall, mAP50, mAP50-95, per-class metrics
    """
    def match_predictions_to_targets(filtered_preds, targets, iou_thresh):
        """
        Match predictions to ground truth targets at a given IoU threshold.

        Returns:
            all_detections: dict mapping class_id to list of (confidence, is_true_positive) tuples
            all_ground_truths: dict mapping class_id to count of ground truth boxes
        """
        all_detections = defaultdict(list)
        all_ground_truths = defaultdict(int)

        for pred, target in zip(filtered_preds, targets):
            pred_boxes = pred['boxes'].cpu()
            pred_labels = pred['labels'].cpu()
            pred_scores = pred['scores'].cpu()

            target_boxes = target['boxes'].cpu()
            target_labels = target['labels'].cpu()

            # Track which ground truth boxes have been matched
            matched_gt = set()

            # Sort predictions by confidence (descending)
            if len(pred_scores) > 0:
                sorted_indices = torch.argsort(pred_scores, descending=True)
                pred_boxes = pred_boxes[sorted_indices]
                pred_labels = pred_labels[sorted_indices]
                pred_scores = pred_scores[sorted_indices]

            # Count ground truth boxes per class
            for label in target_labels:
                all_ground_truths[label.item()] += 1

            # Match predictions to ground truths
            for pred_box, pred_label, pred_score in zip(pred_boxes, pred_labels, pred_scores):
                pred_class = pred_label.item()

                # Find matching ground truth boxes of the same class
                class_mask = target_labels == pred_label
                if class_mask.sum() == 0:
                    # No ground truth of this class -> false positive
                    all_detections[pred_class].append((pred_score.item(), False))
                    continue

                # Compute IoU with all ground truth boxes of the same class
                gt_boxes_same_class = target_boxes[class_mask]
                ious = box_iou(pred_box.unsqueeze(0), gt_boxes_same_class)[0]

                # Find best matching ground truth
                best_iou, best_gt_idx = ious.max(0)

                # Map back to original index
                class_indices = torch.where(class_mask)[0]
                original_gt_idx = class_indices[best_gt_idx].item()

                # Check if this is a true positive
                if best_iou >= iou_thresh and original_gt_idx not in matched_gt:
                    all_detections[pred_class].append((pred_score.item(), True))
                    matched_gt.add(original_gt_idx)
                else:
                    all_detections[pred_class].append((pred_score.item(), False))

        return all_detections, all_ground_truths

    def compute_ap_from_detections(all_detections, all_ground_truths, num_classes):
        """Compute AP for all classes from detection results."""
        total_ap = 0.0
        num_classes_present = 0

        for class_id in range(1, num_classes):  # Skip background (class 0)
            detections = all_detections.get(class_id, [])
            num_gt = all_ground_truths.get(class_id, 0)

            if num_gt == 0:
                # No ground truth for this class
                continue

            num_classes_present += 1

            # Sort by confidence
            detections = sorted(detections, key=lambda x: x[0], reverse=True)

            # Compute precision and recall at each threshold
            tp_cumsum = 0
            fp_cumsum = 0
            precisions = []
            recalls = []

            for conf, is_tp in detections:
                if is_tp:
                    tp_cumsum += 1
                else:
                    fp_cumsum += 1

                precision = tp_cumsum / (tp_cumsum + fp_cumsum) if (tp_cumsum + fp_cumsum) > 0 else 0
                recall = tp_cumsum / num_gt if num_gt > 0 else 0

                precisions.append(precision)
                recalls.append(recall)

            # Compute AP using 11-point interpolation
            if len(precisions) > 0:
                recall_thresholds = np.linspace(0, 1, 11)
                interpolated_precisions = []
                for r_threshold in recall_thresholds:
                    valid_precisions = [p for p, r in zip(precisions, recalls) if r >= r_threshold]
                    if valid_precisions:
                        interpolated_precisions.append(max(valid_precisions))
                    else:
                        interpolated_precisions.append(0.0)
                ap = np.mean(interpolated_precisions)
            else:
                ap = 0.0

            total_ap += ap

        # Average AP across classes
        if num_classes_present > 0:
            return total_ap / num_classes_present, num_classes_present
        else:
            return 0.0, 0

    # Filter predictions by confidence
    filtered_preds = []
    for pred in predictions:
        mask = pred['scores'] >= conf_threshold
        filtered_preds.append({
            'boxes': pred['boxes'][mask],
            'labels': pred['labels'][mask],
            'scores': pred['scores'][mask],
        })

    # Compute mAP50-95 (average over IoU thresholds 0.5:0.05:0.95)
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    ap_values = []

    for iou_thresh in iou_thresholds:
        detections, ground_truths = match_predictions_to_targets(filtered_preds, targets, iou_thresh)
        ap, _ = compute_ap_from_detections(detections, ground_truths, num_classes)
        ap_values.append(ap)

    mAP50 = ap_values[0]  # First IoU threshold is 0.5
    mAP50_95 = np.mean(ap_values)

    # Compute final precision, recall, and per-class metrics at IoU=0.5
    all_detections, all_ground_truths = match_predictions_to_targets(filtered_preds, targets, iou_threshold)

    # Compute per-class metrics
    class_metrics = {}
    total_precision = 0.0
    total_recall = 0.0
    classes_found = 0

    for class_id in range(1, num_classes):
        detections = all_detections.get(class_id, [])
        num_gt = all_ground_truths.get(class_id, 0)

        if num_gt == 0:
            continue

        classes_found += 1
        detections = sorted(detections, key=lambda x: x[0], reverse=True)

        tp_cumsum = 0
        fp_cumsum = 0
        precisions = []
        recalls = []

        for conf, is_tp in detections:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1

            precision = tp_cumsum / (tp_cumsum + fp_cumsum) if (tp_cumsum + fp_cumsum) > 0 else 0
            recall = tp_cumsum / num_gt if num_gt > 0 else 0

            precisions.append(precision)
            recalls.append(recall)

        if len(precisions) > 0:
            final_precision = precisions[-1]
            final_recall = recalls[-1]

            # Compute AP50 for this class
            recall_thresholds = np.linspace(0, 1, 11)
            interpolated_precisions = []
            for r_threshold in recall_thresholds:
                valid_precisions = [p for p, r in zip(precisions, recalls) if r >= r_threshold]
                if valid_precisions:
                    interpolated_precisions.append(max(valid_precisions))
                else:
                    interpolated_precisions.append(0.0)
            ap50 = np.mean(interpolated_precisions)
        else:
            final_precision = 0.0
            final_recall = 0.0
            ap50 = 0.0

        class_metrics[class_id] = {
            'precision': final_precision,
            'recall': final_recall,
            'ap50': ap50,
        }

        total_precision += final_precision
        total_recall += final_recall

    # Average metrics across classes
    if classes_found > 0:
        avg_precision = total_precision / classes_found
        avg_recall = total_recall / classes_found
    else:
        avg_precision = 0.0
        avg_recall = 0.0

    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'mAP50': mAP50,
        'mAP50_95': mAP50_95,
        'class_metrics': class_metrics,
        'num_classes_present': len([c for c in range(1, num_classes) if all_ground_truths.get(c, 0) > 0]),
    }


# ==============================================================================
# OUTPUT AND CHECKPOINTING
# ==============================================================================

def find_last_checkpoint(checkpoint_dir: Path) -> Path:
    """
    Find the most recent checkpoint in the weights directory.

    Args:
        checkpoint_dir: Run directory containing weights subdirectory

    Returns:
        Path to the last checkpoint, or None if no checkpoints found
    """
    # Look in the weights subdirectory
    weights_dir = checkpoint_dir / "weights"

    if not weights_dir.exists():
        return None

    # First try to find 'last.pt'
    last_checkpoint = weights_dir / "last.pt"
    if last_checkpoint.exists():
        return last_checkpoint

    # Otherwise, find the checkpoint with highest epoch number
    epoch_checkpoints = list(weights_dir.glob("epoch*.pt"))
    if epoch_checkpoints:
        # Extract epoch numbers and find max
        def get_epoch_num(path: Path) -> int:
            try:
                # Extract number from "epochX.pt"
                return int(path.stem.replace('epoch', ''))
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


def save_checkpoint(checkpoint_path: Path, epoch: int, model, optimizer, lr_scheduler,
                   best_val_loss: float, metrics_history: dict, config: dict, ema=None):
    """
    Save training checkpoint.

    Args:
        checkpoint_path: Path to save checkpoint
        epoch: Current epoch
        model: Model instance
        optimizer: Optimizer instance
        lr_scheduler: Learning rate scheduler instance
        best_val_loss: Best validation loss so far
        metrics_history: Training metrics history
        config: Configuration dict
        ema: ModelEMA instance (optional)
    """
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

    torch.save(checkpoint_data, checkpoint_path)


def save_results_summary(results_file: Path, config_path: str, metrics_history: dict, best_val_loss: float):
    """
    Save final training results summary.

    Args:
        results_file: Path to results file
        config_path: Path to config file
        metrics_history: Training metrics history
        best_val_loss: Best validation loss
    """
    with open(results_file, 'w') as f:
        f.write("Faster R-CNN Training Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Epochs: {len(metrics_history['train_loss'])}\n")
        f.write(f"Best Validation Loss: {best_val_loss:.4f}\n")
        f.write(f"Final Training Loss: {metrics_history['train_loss'][-1]:.4f}\n")
        if metrics_history['val_loss']:
            f.write(f"Final Validation Loss: {metrics_history['val_loss'][-1]:.4f}\n")
        if metrics_history['val_mAP50']:
            f.write(f"Best mAP50: {max(metrics_history['val_mAP50']):.4f}\n")
            f.write(f"Final mAP50: {metrics_history['val_mAP50'][-1]:.4f}\n")
        if metrics_history['val_precision']:
            f.write(f"Final Precision: {metrics_history['val_precision'][-1]:.4f}\n")
        if metrics_history['val_recall']:
            f.write(f"Final Recall: {metrics_history['val_recall'][-1]:.4f}\n")


def export_onnx(model, config: dict, weights_dir: Path, device, logger):
    """
    Export best model to ONNX format.

    Args:
        model: Model instance
        config: Configuration dict
        weights_dir: Path to weights directory
        device: Device (cuda/cpu)
        logger: Logger instance
    """
    try:
        logger.info("=" * 80)
        logger.info("Exporting best model to ONNX...")

        # Load best model
        best_checkpoint_path = weights_dir / "best.pt"
        if best_checkpoint_path.exists():
            checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            # Create dummy input for ONNX export
            dummy_input = torch.randn(1, 1, config['model']['img_size'], config['model']['img_size']).to(device)

            # Export path
            onnx_path = weights_dir / "best.onnx"
            opset_version = config['export'].get('onnx_opset', 14)

            logger.info(f"Exporting to: {onnx_path}")
            logger.info(f"ONNX opset version: {opset_version}")

            # Export model
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['images'],
                output_names=['output'],
                dynamic_axes={
                    'images': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )

            logger.info(f"✓ ONNX export successful: {onnx_path}")
        else:
            logger.warning(f"Best checkpoint not found: {best_checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to export ONNX: {e}")
        logger.warning("Continuing without ONNX export...")


# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================

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
        epochs: Number of epochs. If None, uses config value.
        batch_size: Batch size. If None, uses config value.
        resume: Resume from last training. Default: False
        checkpoint_path: Path to checkpoint to resume from
        config_path: Path to config YAML file
        name: Name for this training run (default: 'train')
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
            # Extract run directory from checkpoint path (go up to parent of weights/)
            run_dir = Path(checkpoint_path).parent.parent
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

    # Create weights subdirectory for checkpoint files
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

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

    # Cache images in RAM for faster loading (optional, uses more memory)
    cache_images = config['data'].get('cache_images', False)
    if cache_images:
        logger.info("Image caching enabled - loading all images into RAM...")

    train_dataset = CAMELFasterRCNNDataset(
        root_dir=data_dir,
        split='train',
        augment=True,
        config=config,
        cache_images=cache_images
    )

    val_dataset = CAMELFasterRCNNDataset(
        root_dir=data_dir,
        split='val',
        augment=False,
        config=config,
        cache_images=cache_images
    )

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    # Create dataloaders with optimizations
    use_pin_memory = config['data']['pin_memory'] and torch.cuda.is_available()
    num_workers = config['data']['num_workers']

    # Enable persistent workers to avoid worker restart overhead
    persistent_workers = num_workers > 0
    # Prefetch batches to reduce data loading bottleneck
    prefetch_factor = config['data'].get('prefetch_factor', 2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=use_pin_memory,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=use_pin_memory,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    # Build model
    logger.info("Building model...")
    num_classes = config['data']['num_classes']
    model = build_faster_rcnn_model(num_classes, config)
    model.to(device)
    logger.info(f"Model: Faster R-CNN with MobileNetV3 Large ({num_classes} classes)")

    # Optimize model with torch.compile (PyTorch 2.0+)
    if config['model'].get('compile', False):
        try:
            logger.info("Compiling model with torch.compile for better performance...")
            model = torch.compile(model, mode='reduce-overhead')
            logger.info("✓ Model compilation successful")
        except Exception as e:
            logger.warning(f"torch.compile not available or failed: {e}")
            logger.warning("Continuing without compilation...")

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=config['training']['learning_rate'],
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay']
    )

    # Learning rate scheduler with warmup
    warmup_epochs = config['training'].get('warmup_epochs', 3)
    total_epochs = epochs

    # Cosine annealing scheduler (without warmup)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=config['training']['learning_rate'] * 0.01  # Final LR is 1% of initial
    )

    logger.info(f"Learning Rate Scheduler: CosineAnnealingLR with {warmup_epochs} epoch warmup")
    logger.info(f"Initial LR: {config['training']['learning_rate']}, Final LR: {config['training']['learning_rate'] * 0.01}")

    # Setup EMA (Exponential Moving Average) if enabled
    ema = None
    if config['training'].get('ema', False):
        ema_tau = config['training'].get('ema_tau', 0.999)
        ema = ModelEMA(model, tau=ema_tau)
        logger.info(f"EMA enabled with tau={ema_tau}")

    # Checkpoint directory is the run directory
    checkpoint_dir = run_dir
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"Weights directory: {weights_dir}")

    # Save config to run directory for reproducibility (YOLO-style naming)
    config_save_path = run_dir / "args.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Config saved to: {config_save_path}")

    # Setup CSV file for YOLO-style results export
    csv_file = run_dir / "results.csv"
    csv_exists = csv_file.exists() and (resume or checkpoint_path)

    if not csv_exists:
        # Create new CSV with headers
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch',
                'time',
                'train/cls_loss',
                'train/box_loss',
                'train/objectness_loss',
                'train/rpn_box_loss',
                'metrics/precision',
                'metrics/recall',
                'metrics/mAP50',
                'metrics/mAP50-95',
                'val/cls_loss',
                'val/box_loss',
                'val/objectness_loss',
                'val/rpn_box_loss',
                'lr',
            ])
        logger.info(f"CSV file created: {csv_file}")
    else:
        logger.info(f"CSV file exists, will append: {csv_file}")

    # Track cumulative training time
    cumulative_time = 0.0

    # Initialize training state
    start_epoch = 0
    best_val_loss = float('inf')
    metrics_history = {
        'train_loss': [],
        'train_loss_classifier': [],
        'train_loss_box_reg': [],
        'train_loss_objectness': [],
        'train_loss_rpn_box_reg': [],
        'val_loss': [],
        'val_loss_classifier': [],
        'val_loss_box_reg': [],
        'val_loss_objectness': [],
        'val_loss_rpn_box_reg': [],
        'val_precision': [],
        'val_recall': [],
        'val_mAP50': [],
        'val_mAP50_95': [],
    }

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

            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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

            # Estimate cumulative time
            cumulative_time = len(metrics_history.get('train_loss', [])) * 120.0

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
    last_epoch = start_epoch - 1

    for epoch in range(start_epoch, epochs):
        last_epoch = epoch
        epoch_start_time = time.time()

        # Learning rate warmup (linear warmup for first warmup_epochs)
        if epoch < warmup_epochs:
            # Linear warmup: lr = base_lr * (epoch + 1) / warmup_epochs
            warmup_factor = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = config['training']['learning_rate'] * warmup_factor
            logger.info(f"Warmup: Epoch {epoch+1}/{warmup_epochs}, LR = {optimizer.param_groups[0]['lr']:.6f}")

        # Train
        train_losses = train_epoch(model, train_loader, optimizer, device, config, epoch, logger, ema=ema)
        metrics_history['train_loss'].append(train_losses['total_loss'])
        metrics_history['train_loss_classifier'].append(train_losses['loss_classifier'])
        metrics_history['train_loss_box_reg'].append(train_losses['loss_box_reg'])
        metrics_history['train_loss_objectness'].append(train_losses['loss_objectness'])
        metrics_history['train_loss_rpn_box_reg'].append(train_losses['loss_rpn_box_reg'])

        # Initialize validation metrics (in case validation doesn't run this epoch)
        val_metrics = {
            'total_loss': 0.0,
            'loss_classifier': 0.0,
            'loss_box_reg': 0.0,
            'loss_objectness': 0.0,
            'loss_rpn_box_reg': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'mAP50': 0.0,
            'mAP50_95': 0.0,
        }

        # Validate
        if (epoch + 1) % config['validation']['val_interval'] == 0:
            val_metrics = validate(model, val_loader, device, config, epoch, logger, ema=ema)
            val_loss = val_metrics['total_loss']

            # Save validation metrics
            metrics_history['val_loss'].append(val_loss)
            metrics_history['val_loss_classifier'].append(val_metrics['loss_classifier'])
            metrics_history['val_loss_box_reg'].append(val_metrics['loss_box_reg'])
            metrics_history['val_loss_objectness'].append(val_metrics['loss_objectness'])
            metrics_history['val_loss_rpn_box_reg'].append(val_metrics['loss_rpn_box_reg'])
            metrics_history['val_precision'].append(val_metrics['precision'])
            metrics_history['val_recall'].append(val_metrics['recall'])
            metrics_history['val_mAP50'].append(val_metrics['mAP50'])
            metrics_history['val_mAP50_95'].append(val_metrics['mAP50_95'])

        # Calculate epoch time and update cumulative time
        epoch_time = time.time() - epoch_start_time
        cumulative_time += epoch_time

        # Write to CSV file (YOLO-style format)
        current_lr = optimizer.param_groups[0]['lr']
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                f'{cumulative_time:.3f}',
                f'{train_losses["loss_classifier"]:.5f}',
                f'{train_losses["loss_box_reg"]:.5f}',
                f'{train_losses["loss_objectness"]:.5f}',
                f'{train_losses["loss_rpn_box_reg"]:.5f}',
                f'{val_metrics["precision"]:.5f}',
                f'{val_metrics["recall"]:.5f}',
                f'{val_metrics["mAP50"]:.5f}',
                f'{val_metrics["mAP50_95"]:.5f}',
                f'{val_metrics["loss_classifier"]:.5f}',
                f'{val_metrics["loss_box_reg"]:.5f}',
                f'{val_metrics["loss_objectness"]:.5f}',
                f'{val_metrics["loss_rpn_box_reg"]:.5f}',
                f'{current_lr:.6f}',
            ])

        # Early stopping (validation runs, check if val_loss improved)
        if (epoch + 1) % config['validation']['val_interval'] == 0:
            val_loss = val_metrics['total_loss']
            # Early stopping (only if val_loss is valid, not NaN)
            if not np.isnan(val_loss) and val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save best model
                if config['checkpoint'].get('save_best', True):
                    best_checkpoint_path = weights_dir / "best.pt"
                    save_checkpoint(best_checkpoint_path, epoch, model, optimizer, lr_scheduler,
                                  best_val_loss, metrics_history, config, ema)
                    logger.info(f"Best model saved: {best_checkpoint_path} (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1

            if patience_counter >= max_patience:
                logger.info(f"Early stopping triggered after {max_patience} epochs without improvement")
                break

        # Save checkpoint at intervals
        if (epoch + 1) % config['checkpoint']['save_interval'] == 0:
            interval_checkpoint_path = weights_dir / f"epoch{epoch+1}.pt"
            save_checkpoint(interval_checkpoint_path, epoch, model, optimizer, lr_scheduler,
                          best_val_loss, metrics_history, config, ema)
            logger.info(f"Checkpoint saved: {interval_checkpoint_path}")

        # Learning rate scheduling (only after warmup)
        if epoch >= warmup_epochs:
            lr_scheduler.step()

        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

    # Save last model
    if config['checkpoint'].get('save_last', True):
        last_checkpoint_path = weights_dir / "last.pt"
        save_checkpoint(last_checkpoint_path, last_epoch, model, optimizer, lr_scheduler,
                       best_val_loss, metrics_history, config, ema)
        logger.info(f"Last model saved: {last_checkpoint_path}")

    # Save metrics to run directory
    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics_history, f, indent=2)
    logger.info(f"Metrics saved to: {metrics_file}")
    logger.info(f"CSV results saved to: {csv_file}")

    # Save final results summary
    results_file = run_dir / "results.txt"
    save_results_summary(results_file, config_path, metrics_history, best_val_loss)
    logger.info(f"Results summary saved to: {results_file}")

    # Export best model to ONNX
    if config['export'].get('onnx', True):
        export_onnx(model, config, weights_dir, device, logger)

    logger.info("Training completed!")
    logger.info("=" * 80)


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

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
