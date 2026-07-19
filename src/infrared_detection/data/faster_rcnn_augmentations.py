"""
Albumentations-based augmentations for Faster R-CNN training on infrared images.E
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Dict, Tuple
import cv2


class FasterRCNNAugmentations:
    """
    Augmentation pipeline for Faster R-CNN training on infrared CAMEL dataset.
    
    Applies geometric and color augmentations with bounding box handling.
    Images are kept in [0, 1] range (no standardization) to match YOLOv8 preprocessing.
    
    Args:
        config (dict): Augmentation configuration from faster_rcnn_config.yaml
        img_size (tuple): Image size (height, width), e.g., (256, 336)
    """
    
    def __init__(self, config: dict, img_size: Tuple[int, int] = (256, 336)):
        self.config = config
        self.height, self.width = img_size
        
        # Build augmentation pipeline
        self.transform = self._build_augmentation_pipeline()
    
    def _build_augmentation_pipeline(self) -> A.Compose:
        """Build albumentations augmentation pipeline"""
        
        transforms = []

        fill_value = 0.4  # A dark gray value in [0, 255] scale (Albumentations handles conversion for float32)

        # 1. COLOR/BRIGHTNESS AUGMENTATIONS (approximating HSV adjustments)

        # YOLOv8: hsv_s=0.7, hsv_v=0.4 (always applied)
        # For grayscale infrared: use brightness/contrast to approximate HSV
        
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=self.config.get('brightness_limit', 0.4),
                contrast_limit=self.config.get('contrast_limit', 0.4),
                p=1.0  # Always apply (YOLOv8 behavior)
            )
        )

        # 2. GEOMETRIC AUGMENTATIONS
        
        # Rotation + Translation + Scale combined
        # Defaults: degrees=15.0, translate=0.1, scale=0.5-
        transforms.append(
            A.Affine(
                translate_percent={'x': (-self.config.get('translation_limit', 0.1), self.config.get('translation_limit', 0.1)),
                                   'y': (-self.config.get('translation_limit', 0.1), self.config.get('translation_limit', 0.1))},
                scale=(1.0 - self.config.get('scale_limit', 0.5), 1.0 + self.config.get('scale_limit', 0.5)),
                rotate=(-int(self.config.get('degrees', 15.0)), int(self.config.get('degrees', 15.0))),
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                fill=fill_value,  # Fill with a dark gray value to avoid pure black holes
                p=1.0  # Always apply rotation (Similar to YOLOv8 behavior)
            )
        )
        
        # Horizontal flip
        # YOLOv8: fliplr=0.5
        if self.config.get('horizontal_flip_p', 0.5) > 0:
            transforms.append(
                A.HorizontalFlip(
                    p=self.config.get('horizontal_flip_p', 0.5)
                )
            )
        
        # Vertical flip
        # YOLOv8: flipud=0.0 (typically not used for natural images, but we can experiment)
        if self.config.get('vertical_flip_p', 0.0) > 0:
            transforms.append(
                A.VerticalFlip(
                    p=self.config.get('vertical_flip_p', 0.0)
                )
            )
        
        # 3. CUTOUT/ERASING AUGMENTATION
        # YOLOv8: erasing=0.4
        
        if self.config.get('cutout_enabled', True):
            max_holes = self.config.get('cutout_max_holes', 4)
            max_h = self.config.get('cutout_max_height', 20)
            max_w = self.config.get('cutout_max_width', 20)
            transforms.append(
                A.CoarseDropout(
                    num_holes_range=(1, max_holes),
                    hole_height_range=(max_h, max_h),  # Fixed size holes
                    hole_width_range=(max_w, max_w),   # Fixed size holes
                    fill=fill_value,  # Fill with a dark gray value to avoid pure black holes
                    fill_mask=None,
                    p=self.config.get('cutout_p', 0.4)
                )
            )
        
        # 4. CONVERT TO TENSOR
        # Before ToTensorV2: np.ndarray shape (256, 336, 1), dtype float32
        # After ToTensorV2: torch.Tensor shape (1, 256, 336), dtype float32
        
        transforms.append(ToTensorV2())
        
        # Create Compose with bbox params for proper label transformation
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format='pascal_voc',  # (x_min, y_min, x_max, y_max) in pixels
                label_fields=['class_labels'],
                min_visibility=0.2,  # Keep boxes with at least 20% visibility
                min_area=16.0,  # Minimum bbox area in pixels (e.g., 4x4 pixels)
                clip=True,  # Clip bboxes to image boundaries
            )
        )
    
    def __call__(self, image: np.ndarray, bboxes: list, class_ids: list) -> Tuple[np.ndarray, list, list]:
        """
        Apply augmentations to image and bounding boxes.
        
        Args:
            image (np.ndarray): Input image, shape (H, W) or (H, W, C), values in [0, 1]
            bboxes (list): List of bounding boxes in Pascal VOC format (x_min, y_min, x_max, y_max) in pixels
            class_ids (list): List of class IDs for each bbox
        
        Returns:
            Tuple of (augmented_image, augmented_bboxes, augmented_class_ids)
        """
        
        # Ensure image is in correct format
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)  # (H, W) → (H, W, 1)
        
        # Ensure image is float32 in [0, 1]
        if image.dtype != np.float32:
            if image.max() > 1.0:
                image = image.astype(np.float32) / 255.0
            else:
                image = image.astype(np.float32)
        
        image = np.clip(image, 0.0, 1.0)
        
        # Validate bounding boxes (Pascal VOC format: x_min, y_min, x_max, y_max)
        valid_bboxes = []
        valid_class_ids = []
        for bbox, cls_id in zip(bboxes, class_ids):
            if len(bbox) == 4:
                x_min, y_min, x_max, y_max = bbox
                # Basic sanity check
                if x_max > x_min and y_max > y_min:
                    valid_bboxes.append(bbox)
                    valid_class_ids.append(cls_id)
        
        # Apply augmentation (albumentations handles bbox transformation)
        if valid_bboxes:
            try:
                augmented = self.transform(
                    image=image,
                    bboxes=valid_bboxes,
                    class_labels=valid_class_ids
                )
                augmented_image = augmented['image']
                augmented_bboxes = augmented['bboxes']
                augmented_class_ids = augmented['class_labels']
            except (ValueError, AssertionError) as e:
                # If transformation still fails, return original image with valid boxes
                print(f"Warning: Albumentations transform failed: {e}")
                print(f"         Returning image without augmentation")
                # Convert to tensor manually
                from albumentations.pytorch import ToTensorV2
                tensor_transform = ToTensorV2()
                augmented_image = tensor_transform(image=image)['image']
                augmented_bboxes = valid_bboxes
                augmented_class_ids = valid_class_ids
        else:
            # No valid bboxes case
            augmented = self.transform(image=image, bboxes=[], class_labels=[])
            augmented_image = augmented['image']
            augmented_bboxes = []
            augmented_class_ids = []
        
        return augmented_image, augmented_bboxes, augmented_class_ids


class InferenceAugmentations:
    """
    Minimal augmentations for inference (no geometric transforms).
    Images are kept in [0, 1] range (no standardization) to match YOLOv8 preprocessing.
    This is necessary as the inout labels are formatted for YOLO and the model 
    expects the same format during inference.

    TODO: Replace with C++ implementation for faster inference and to avoid unnecessary
    conversions between numpy and torch tensors.
    """
    
    def __init__(self, img_size: Tuple[int, int] = (256, 336)):
        self.height, self.width = img_size
        
        transforms = []
        
        # No standardization - images remain in [0, 1] range for fair comparison with YOLOv8
        transforms.append(ToTensorV2())
        
        # No bbox_params needed since ToTensorV2 doesn't modify bounding boxes
        self.transform = A.Compose(transforms)
    
    def __call__(self, image: np.ndarray, bboxes: list = None, class_ids: list = None):
        """Apply inference augmentations (tensor conversion only)."""
        
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        if image.dtype != np.float32:
            if image.max() > 1.0:
                image = image.astype(np.float32) / 255.0
            else:
                image = image.astype(np.float32)
        
        image = np.clip(image, 0.0, 1.0)
        
        if bboxes is None:
            bboxes = []
        if class_ids is None:
            class_ids = []
        
        # Only transform image (no bbox processing in inference)
        augmented = self.transform(image=image)
        
        return augmented['image'], bboxes, class_ids
