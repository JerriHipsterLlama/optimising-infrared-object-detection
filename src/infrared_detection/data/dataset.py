"""
CAMEL Dataset loader for infrared object detection.

Loads 336x256 grayscale infrared images and their corresponding Pascal VOC format labels.
Supports train/val/test splits.

This version loads labels directly in corner format (x1, y1, x2, y2) from 
labels_pascal directory, eliminating conversion overhead and potential sources of error.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from infrared_detection.data.transforms import ImageTransforms, InferenceTransforms


class CAMELDataset(Dataset):
    """
    PyTorch Dataset for CAMEL infrared imagery with Pascal VOC format labels.
    
    Dataset structure:
    - data/camel/images/train/Seq##_*.png      (infrared images, 336x256)
    - data/camel/labels_pascal/train/Seq##_*.txt  (Pascal VOC format labels)
    
    Pascal VOC label format per line: <class_id> <x1> <y1> <x2> <y2>
    Coordinates are in pixels (corner format) - no conversion needed!
    
    Args:
        root_dir (str): Path to data/camel directory
        split (str): 'train', 'val', or 'test'. Default: 'train'
        augment (bool): Apply augmentations (only for training). Default: True
        normalize (bool): Normalize pixel values. Default: True
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        augment: bool = True,
        normalize: bool = True,
    ):
        """Initialize CAMEL dataset with Pascal VOC format labels."""
        self.root_dir = Path(root_dir)
        self.split = split
        
        self.images_dir = self.root_dir / "images" / split
        pascal_labels_dir = self.root_dir / "labels_pascal" / split
        yolo_labels_dir = self.root_dir / "labels" / split
        self.labels_format = "pascal" if pascal_labels_dir.exists() else "yolo"
        self.labels_dir = pascal_labels_dir if pascal_labels_dir.exists() else yolo_labels_dir
        
        # Verify directories exist
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(
                f"Labels directory not found: {self.labels_dir}\n"
                f"Run: python tools/convert_labels_to_pascal_format.py --split {split}"
            )
        
        # Image properties (CAMEL infrared standard size)
        self.image_height = 256
        self.image_width = 336
        
        # Load image paths (support multiple formats)
        self.image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.npy']:
            self.image_files.extend(sorted(self.images_dir.glob(f"Seq*{ext}")))
        self.image_files = sorted(set(self.image_files))  # Remove duplicates
        
        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in {self.images_dir}")
        
        self.image_to_labels = {
            str(path): self._label_path_for_image(path)
            for path in self.image_files
        }

        print(f"Loaded {len(self.image_files)} images from {split} split ({self.labels_format} format)")
        
        # Set up transforms
        if split == "train" and augment:
            self.transforms = ImageTransforms(normalize=normalize)
        else:
            self.transforms = InferenceTransforms(normalize=normalize)
    
    def __len__(self) -> int:
        """Return the total number of images in the dataset."""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Load image and corresponding labels.
        
        Args:
            idx (int): Index of image to load
        
        Returns:
            Tuple of:
                - image (torch.Tensor): Shape (1, 256, 336), normalized to [0, 1]
                - targets (Dict): Dictionary containing:
                    - 'boxes': torch.Tensor of shape (N, 4) in corner format (x1, y1, x2, y2)
                    - 'class_ids': torch.Tensor of shape (N,)
                    - 'image_path': str, path to image file
        """
        img_path = self.image_files[idx]
        
        # Load image
        image = self._load_image(img_path)
        
        # Apply transformations
        image = self.transforms(image)
        
        # Load labels and normalize them to corner coordinates.
        targets = self._load_labels(img_path)
        
        return image, targets
    
    def _load_image(self, img_path: Path) -> np.ndarray:
        """
        Load infrared image from disk.
        
        Args:
            img_path (Path): Path to image file
        
        Returns:
            np.ndarray: Image array, shape (H, W), pixel values as original
        """
        if img_path.suffix == '.npy':
            # Load numpy array directly
            image = np.load(img_path)
        else:
            # Load image file
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Normalize to [0, 1] if needed
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        return image
    
    def _load_labels(self, img_path: Path) -> Dict:
        """
        Load Pascal VOC format labels for the image.
        
        Format: class_id x1 y1 x2 y2 (corner coordinates in pixels)
        No conversion needed - labels are already in the correct format!
        
        Args:
            img_path (Path): Path to image file
        
        Returns:
            Dict containing:
                - 'boxes': torch.Tensor of shape (N, 4) - corner format (x1, y1, x2, y2)
                - 'class_ids': torch.Tensor of shape (N,)
                - 'image_path': str
        """
        label_file = self._label_path_for_image(img_path)
        
        boxes = []
        class_ids = []
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            # Pascal: class x1 y1 x2 y2; YOLO: class cx cy width height.
                            class_id = int(parts[0])
                            values = [float(value) for value in parts[1:5]]
                            if self.labels_format == "yolo":
                                cx, cy, width, height = values
                                x1 = (cx - width / 2.0) * self.image_width
                                y1 = (cy - height / 2.0) * self.image_height
                                x2 = (cx + width / 2.0) * self.image_width
                                y2 = (cy + height / 2.0) * self.image_height
                            else:
                                x1, y1, x2, y2 = values
                            
                            # Validate box (ensure x2 > x1 and y2 > y1)
                            if x2 > x1 and y2 > y1:
                                boxes.append([x1, y1, x2, y2])
                                class_ids.append(class_id)
                        except (ValueError, IndexError):
                            continue
        
        # Convert to tensors
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            class_ids = torch.tensor(class_ids, dtype=torch.long)
        else:
            # Empty tensors for images with no objects (background class)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            class_ids = torch.zeros((0,), dtype=torch.long)
        
        targets = {
            'boxes': boxes,
            'class_ids': class_ids,
            'image_path': str(img_path),
        }
        
        return targets

    def _label_path_for_image(self, img_path: Path) -> Path:
        """Resolve either per-image labels or the legacy per-sequence label."""

        per_image = self.labels_dir / f"{img_path.stem}.txt"
        if per_image.exists():
            return per_image
        sequence_name = img_path.stem.split("_")[0]
        return self.labels_dir / f"{sequence_name}.txt"
    
    def get_image_info(self, idx: int) -> Dict:
        """
        Get metadata about an image without loading it.
        
        Args:
            idx (int): Image index
        
        Returns:
            Dict with image path, size, and label info
        """
        img_path = self.image_files[idx]
        label_file = self._label_path_for_image(img_path)
        
        # Count objects in label file
        num_objects = 0
        if label_file.exists():
            with open(label_file, 'r') as f:
                num_objects = sum(1 for line in f if line.strip())
        
        info = {
            'index': idx,
            'image_path': str(img_path),
            'sequence': img_path.stem.split("_")[0],
            'image_size': (self.image_height, self.image_width),
            'has_labels': label_file.exists(),
            'num_objects': num_objects,
        }
        
        return info


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    augment: bool = True,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders with Pascal VOC format labels.
    
    Args:
        data_dir (str): Path to data/camel directory
        batch_size (int): Batch size. Default: 32
        num_workers (int): Number of data loading workers. Default: 4
        augment (bool): Enable augmentation for training. Default: True
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Training dataset with augmentation
    train_dataset = CAMELDataset(
        root_dir=data_dir,
        split='train',
        augment=augment,
    )
    
    # Validation dataset without augmentation
    val_dataset = CAMELDataset(
        root_dir=data_dir,
        split='val',
        augment=False,
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )
    
    return train_loader, val_loader
