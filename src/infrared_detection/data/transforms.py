"""
Image transformations optimized for 336x256 infrared images.
Handles normalization and conversion to PyTorch tensors.

Augmentation is delegated to Ultralytics YOLOv8's built-in pipeline,
which automatically handles label transformation for geometric augmentations.
"""

import numpy as np
import torch
from typing import Optional


class ImageTransforms:
    """
    Minimal transformations for infrared images from CAMEL dataset.
    
    Only handles normalization and tensor conversion.
    Augmentation is handled by Ultralytics YOLOv8 during training.
    
    Args:
        normalize (bool): Normalize pixel values to [0, 1]. Default: True
    """
    
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
    
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """
        Apply transformations to an infrared image.
        
        Args:
            image (np.ndarray): Input image, shape (H, W) or (H, W, C)
        
        Returns:
            torch.Tensor: Transformed image tensor, shape (C, H, W)
        """
        # 1. NORMALIZATION
        if self.normalize:
            image = self._normalize(image)
        
        # 2. ENSURE CORRECT SHAPE AND CONTIGUOUS
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)  # (H, W) → (H, W, 1)
        
        image = np.ascontiguousarray(image)
        
        # 3. CONVERT TO TENSOR (C, H, W)
        tensor: torch.Tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        
        return tensor
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize infrared pixel values to [0, 1].
        
        Args:
            image (np.ndarray): Input image (H, W) uint8 or uint16
        
        Returns:
            np.ndarray: Normalized image in range [0, 1], dtype float32
        """
        if image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0
        elif image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            # Already float
            if image.max() > 1.0:
                image = image / 255.0
        
        return np.clip(image, 0.0, 1.0)


class InferenceTransforms:
    """
    Minimal transformations for inference (no augmentation).
    """
    
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
    
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """Apply inference transformations."""
        if self.normalize:
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            elif image.dtype == np.float32 and image.max() > 1.0:
                image = image / 255.0
            image = np.clip(image, 0.0, 1.0)
        
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        image = np.ascontiguousarray(image)
        tensor: torch.Tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        
        return tensor