"""
CAMEL Dataset loader for infrared object detection.

Loads 336x256 grayscale infrared images and their corresponding YOLO format labels.
Supports train/val/test splits.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from .transforms import ImageTransforms, InferenceTransforms


class CAMELDataset(Dataset):
    """
    PyTorch Dataset for CAMEL infrared imagery.
    
    Dataset structure:
    - data/camel/images/train/Seq##_*.png  (infrared images, 336x256)
    - data/camel/labels/train/Seq##.txt    (YOLO format labels per sequence)
    
    YOLO label format per line: <class_id> <x_center> <y_center> <width> <height>
    All coordinates are normalized to [0, 1] relative to image dimensions.
    
    Args:
        root_dir (str): Path to data/camel directory
        split (str): 'train', 'val', or 'test'. Default: 'train'
        augment (bool): Apply augmentations (only for training). Default: True
        normalize (bool): Normalize pixel values. Default: True
        box_format (str): 'yolo' or 'corner'. Default: 'yolo'
            - 'yolo': (x_center, y_center, width, height) normalized [0, 1]
            - 'corner': (x1, y1, x2, y2) in pixels for Faster R-CNN
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        augment: bool = True,
        normalize: bool = True,
        box_format: str = "yolo",
    ):
        """Initialize CAMEL dataset."""
        self.root_dir = Path(root_dir)
        self.split = split
        self.box_format = box_format.lower()
        
        if self.box_format not in ["yolo", "corner"]:
            raise ValueError(f"box_format must be 'yolo' or 'corner', got {self.box_format}")
        
        self.images_dir = self.root_dir / "images" / split
        self.labels_dir = self.root_dir / "labels" / split
        
        # Verify directories exist
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")
        
        # Image properties
        self.image_height = 256
        self.image_width = 336
        
        # Load image paths (support both PNG and JPG formats)
        self.image_files = sorted(self.images_dir.glob("*.png"))
        self.image_files.extend(sorted(self.images_dir.rglob("*.jpg")))
        self.image_files.extend(sorted(self.images_dir.rglob("*.jpeg")))
        self.image_files = sorted(set(self.image_files))  # Remove duplicates and sort
        
        if len(self.image_files) == 0:
            raise ValueError(f"No image files (PNG/JPG) found in {self.images_dir}")
        
        # Load label files (one per sequence)
        self.label_files = sorted(self.labels_dir.glob("*.txt"))
        self._build_image_to_label_mapping()
        
        # Set up transforms
        # Note: Augmentation is delegated to Ultralytics YOLOv8 training pipeline
        # ImageTransforms only handles normalization
        if split == "train":
            self.transforms = ImageTransforms(normalize=normalize)
        else:
            self.transforms = InferenceTransforms(normalize=normalize)
    
    def _build_image_to_label_mapping(self):
        """
        Build a mapping from image files to their corresponding label files.
        
        Labels are stored per-image (Seq##_######.txt), matching the image filename.
        
        Example:
            Image: Seq01_000001.png → Label: Seq01_000001.txt
            Image: Seq05_000042.png → Label: Seq05_000042.txt
        """
        self.image_to_labels = {}
        
        for img_file in self.image_files:
            # Label file has same name as image file but with .txt extension
            label_file = self.labels_dir / f"{img_file.stem}.txt"
            
            if label_file.exists():
                self.image_to_labels[str(img_file)] = str(label_file)
            else:
                print(f"Warning: Label file not found for {img_file}")
    
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
                    - 'boxes': torch.Tensor of shape (N, 4), YOLO format
                    - 'class_ids': torch.Tensor of shape (N,)
                    - 'image_path': str, path to image file
        """
        img_path = self.image_files[idx]
        
        # Load image
        image = self._load_image(img_path)
        
        # Apply transformations
        image = self.transforms(image)
        
        # Load labels
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
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        return image
    
    def _load_labels(self, img_path: Path) -> Dict:
        """
        Load YOLO format labels for the image and convert to requested format.
        
        Args:
            img_path (Path): Path to image file
        
        Returns:
            Dict containing:
                - 'boxes': torch.Tensor of shape (N, 4)
                    * if box_format='yolo': (x_center, y_center, width, height) normalized [0,1]
                    * if box_format='corner': (x1, y1, x2, y2) in pixels
                - 'class_ids': torch.Tensor of shape (N,)
                - 'image_path': str
        """
        # Label file has same name as image file but with .txt extension
        label_file = self.labels_dir / f"{img_path.stem}.txt"
        
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
                        # YOLO format: <class_id> <x_center> <y_center> <width> <height>
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        if self.box_format == "yolo":
                            # Keep as YOLO format (normalized)
                            boxes.append([x_center, y_center, width, height])
                        elif self.box_format == "corner":
                            # Convert to corner format (pixels): x1, y1, x2, y2
                            # Corner format is used by Faster R-CNN
                            x1 = (x_center - width / 2) * self.image_width
                            y1 = (y_center - height / 2) * self.image_height
                            x2 = (x_center + width / 2) * self.image_width
                            y2 = (y_center + height / 2) * self.image_height
                            boxes.append([x1, y1, x2, y2])
                        
                        class_ids.append(class_id)
        
        # Convert to tensors
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            class_ids = torch.tensor(class_ids, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            class_ids = torch.zeros((0,), dtype=torch.long)
        
        targets = {
            'boxes': boxes,
            'class_ids': class_ids,
            'image_path': str(img_path),
        }
        
        return targets
    
    def get_image_info(self, idx: int) -> Dict:
        """
        Get metadata about an image without loading it.
        
        Args:
            idx (int): Image index
        
        Returns:
            Dict with image path, size, and label info
        """
        img_path = self.image_files[idx]
        seq_name = img_path.stem.split("_")[0]
        label_file = self.labels_dir / f"{seq_name}.txt"
        
        info = {
            'index': idx,
            'image_path': str(img_path),
            'sequence': seq_name,
            'image_size': (self.image_height, self.image_width),
            'has_labels': label_file.exists(),
        }
        
        return info


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    augment: bool = True,
    box_format: str = "yolo",
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir (str): Path to data/camel directory
        batch_size (int): Batch size. Default: 32
        num_workers (int): Number of data loading workers. Default: 4
        augment (bool): Enable augmentation for training. Default: True
        box_format (str): 'yolo' or 'corner'. Default: 'yolo'
            - 'yolo': (x_center, y_center, width, height) normalized [0,1]
            - 'corner': (x1, y1, x2, y2) in pixels for Faster R-CNN
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Training dataset with augmentation
    train_dataset = CAMELDataset(
        root_dir=data_dir,
        split='train',
        augment=augment,
        box_format=box_format,
    )
    
    # Validation dataset without augmentation
    val_dataset = CAMELDataset(
        root_dir=data_dir,
        split='val',
        augment=False,
        box_format=box_format,
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
