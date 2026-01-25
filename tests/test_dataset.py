"""
Unit tests for CAMEL dataset loading module.

Tests for:
- CAMELDataset class
- Image loading from disk
- Label parsing from YOLO format
- Dataset splits (train/val/test)
"""

import unittest
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.python.dataset.dataset import CAMELDataset, create_dataloaders


class TestCAMELDatasetInitialization(unittest.TestCase):
    """Test CAMELDataset initialization and basic operations."""
    
    @classmethod
    def setUpClass(cls):
        """Create a temporary test dataset."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.data_dir = Path(cls.temp_dir) / "camel"
        
        # Create directory structure
        cls.train_images = cls.data_dir / "images" / "train"
        cls.train_labels = cls.data_dir / "labels" / "train"
        cls.val_images = cls.data_dir / "images" / "val"
        cls.val_labels = cls.data_dir / "labels" / "val"
        cls.test_images = cls.data_dir / "images" / "test"
        cls.test_labels = cls.data_dir / "labels" / "test"
        
        for d in [cls.train_images, cls.train_labels, cls.val_images, cls.val_labels, 
                  cls.test_images, cls.test_labels]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Create dummy test images (336x256 grayscale)
        for split, img_dir, lbl_dir in [
            ("train", cls.train_images, cls.train_labels),
            ("val", cls.val_images, cls.val_labels),
            ("test", cls.test_images, cls.test_labels),
        ]:
            for seq_idx in range(1, 4):  # 3 sequences
                seq_name = f"Seq{seq_idx:02d}"
                
                # Create 5 images per sequence
                for img_idx in range(5):
                    img_path = img_dir / f"{seq_name}_{img_idx:03d}.png"
                    # Create dummy image
                    img_data = np.random.randint(0, 255, (256, 336), dtype=np.uint8)
                    import cv2
                    cv2.imwrite(str(img_path), img_data)
                
                # Create label file for sequence
                lbl_path = lbl_dir / f"{seq_name}.txt"
                with open(lbl_path, 'w') as f:
                    # Write 3 dummy bounding boxes
                    # Format: class_id x_center y_center width height (normalized)
                    f.write("0 0.5 0.5 0.3 0.4\n")
                    f.write("0 0.2 0.3 0.15 0.2\n")
                    f.write("1 0.8 0.7 0.2 0.25\n")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        shutil.rmtree(cls.temp_dir)
    
    def test_dataset_initialization_train(self):
        """Test dataset initialization with train split."""
        dataset = CAMELDataset(
            root_dir=str(self.data_dir),
            split='train',
            augment=True,
        )
        
        self.assertIsNotNone(dataset)
        self.assertEqual(dataset.split, 'train')
        self.assertEqual(dataset.image_height, 256)
        self.assertEqual(dataset.image_width, 336)
    
    def test_dataset_initialization_val(self):
        """Test dataset initialization with val split."""
        dataset = CAMELDataset(
            root_dir=str(self.data_dir),
            split='val',
            augment=False,
        )
        
        self.assertEqual(dataset.split, 'val')
    
    def test_dataset_length(self):
        """Test that dataset has correct length."""
        dataset = CAMELDataset(
            root_dir=str(self.data_dir),
            split='train',
        )
        
        # 3 sequences × 5 images per sequence = 15 images
        self.assertEqual(len(dataset), 15)
    
    def test_missing_images_directory(self):
        """Test error handling when images directory is missing."""
        bad_path = str(self.data_dir) + "_nonexistent"
        
        with self.assertRaises(FileNotFoundError):
            CAMELDataset(root_dir=bad_path, split='train')
    
    def test_missing_labels_directory(self):
        """Test error handling when labels directory is missing."""
        # Create data dir without labels
        bad_dir = Path(self.temp_dir) / "camel_bad"
        (bad_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        
        with self.assertRaises(FileNotFoundError):
            CAMELDataset(root_dir=str(bad_dir), split='train')


class TestCAMELDatasetLoading(unittest.TestCase):
    """Test data loading and sample retrieval."""
    
    @classmethod
    def setUpClass(cls):
        """Create a temporary test dataset."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.data_dir = Path(cls.temp_dir) / "camel"
        
        # Create directory structure
        cls.train_images = cls.data_dir / "images" / "train"
        cls.train_labels = cls.data_dir / "labels" / "train"
        
        cls.train_images.mkdir(parents=True, exist_ok=True)
        cls.train_labels.mkdir(parents=True, exist_ok=True)
        
        # Create single sequence with known data
        import cv2
        
        seq_name = "Seq01"
        for img_idx in range(3):
            img_path = cls.train_images / f"{seq_name}_{img_idx:03d}.png"
            # Create image with specific values for testing
            img_data = np.full((256, 336), 128, dtype=np.uint8)
            cv2.imwrite(str(img_path), img_data)
        
        # Create label file
        lbl_path = cls.train_labels / f"{seq_name}.txt"
        with open(lbl_path, 'w') as f:
            f.write("0 0.5 0.5 0.3 0.4\n")
            f.write("1 0.2 0.3 0.15 0.2\n")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        shutil.rmtree(cls.temp_dir)
    
    def test_image_to_labels_mapping_direction(self):
        """Ensure image_to_labels maps image_path -> label_path."""
        dataset = CAMELDataset(
            root_dir=str(self.data_dir),
            split='train',
        )
        
        # image_to_labels should not be empty for our synthetic dataset
        self.assertTrue(hasattr(dataset, "image_to_labels"))
        self.assertGreater(len(dataset.image_to_labels), 0)
        
        for img_path_str, lbl_path in dataset.image_to_labels.items():
            # Keys should be image paths inside the train images directory
            self.assertIsInstance(img_path_str, str)
            self.assertTrue(img_path_str.endswith(".png"))
            self.assertIn(str(self.train_images), img_path_str)
            
            # Values should be label paths inside the train labels directory
            self.assertIsInstance(lbl_path, (str, Path))
            self.assertTrue(str(lbl_path).endswith(".txt"))
            self.assertIn(str(self.train_labels), str(lbl_path))
    
    def test_get_item_returns_tuple(self):
        """Test that __getitem__ returns (image, targets) tuple."""
        dataset = CAMELDataset(
            root_dir=str(self.data_dir),
            split='train',
        )
        
        image, targets = dataset[0]
        
        self.assertIsInstance(image, torch.Tensor)
        self.assertIsInstance(targets, dict)
    
    def test_image_tensor_shape(self):
        """Test that loaded image has correct shape."""
        dataset = CAMELDataset(
            root_dir=str(self.data_dir),
            split='train',
        )
        
        image, _ = dataset[0]
        
        # Shape should be (C, H, W) = (1, 256, 336)
        self.assertEqual(len(image.shape), 3)
        self.assertEqual(image.shape[0], 1)  # Channels
        self.assertEqual(image.shape[1], 256)  # Height
        self.assertEqual(image.shape[2], 336)  # Width
    
    def test_image_tensor_dtype(self):
        """Test that loaded image is float32."""
        dataset = CAMELDataset(
            root_dir=str(self.data_dir),
            split='train',
        )
        
        image, _ = dataset[0]
        
        self.assertEqual(image.dtype, torch.float32)
    
    def test_targets_structure(self):
        """Test that targets dict has expected keys."""
        dataset = CAMELDataset(
            root_dir=str(self.data_dir),
            split='train',
        )
        
        _, targets = dataset[0]
        
        self.assertIn('boxes', targets)
        self.assertIn('class_ids', targets)
        self.assertIn('image_path', targets)
    
    def test_boxes_tensor_shape(self):
        """Test that bounding boxes have correct shape."""
        dataset = CAMELDataset(
            root_dir=str(self.data_dir),
            split='train',
        )
        
        _, targets = dataset[0]
        boxes = targets['boxes']
        
        # Should be (N, 4) for N bounding boxes
        self.assertEqual(len(boxes.shape), 2)
        self.assertEqual(boxes.shape[1], 4)  # 4 values per box
        self.assertEqual(boxes.shape[0], 2)  # 2 boxes in test data
    
    def test_class_ids_tensor_type(self):
        """Test that class IDs are integers."""
        dataset = CAMELDataset(
            root_dir=str(self.data_dir),
            split='train',
        )
        
        _, targets = dataset[0]
        class_ids = targets['class_ids']
        
        self.assertEqual(class_ids.dtype, torch.long)
    
    def test_multiple_samples(self):
        """Test loading multiple samples."""
        dataset = CAMELDataset(
            root_dir=str(self.data_dir),
            split='train',
        )
        
        # Load multiple samples
        for idx in range(min(3, len(dataset))):
            image, targets = dataset[idx]
            
            self.assertEqual(image.shape, (1, 256, 336))
            self.assertIn('boxes', targets)
            self.assertIn('class_ids', targets)


class TestCAMELDatasetEmptyLabels(unittest.TestCase):
    """Test handling of images with no bounding boxes."""
    
    @classmethod
    def setUpClass(cls):
        """Create dataset with some empty labels."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.data_dir = Path(cls.temp_dir) / "camel"
        
        cls.train_images = cls.data_dir / "images" / "train"
        cls.train_labels = cls.data_dir / "labels" / "train"
        
        cls.train_images.mkdir(parents=True, exist_ok=True)
        cls.train_labels.mkdir(parents=True, exist_ok=True)
        
        import cv2
        
        # Create images
        seq_name = "Seq01"
        for img_idx in range(2):
            img_path = cls.train_images / f"{seq_name}_{img_idx:03d}.png"
            img_data = np.ones((256, 336), dtype=np.uint8) * 100
            cv2.imwrite(str(img_path), img_data)
        
        # Create empty label file (no bounding boxes)
        lbl_path = cls.train_labels / f"{seq_name}.txt"
        with open(lbl_path, 'w') as f:
            f.write("")  # Empty file
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        shutil.rmtree(cls.temp_dir)
    
    def test_empty_boxes(self):
        """Test that images with no boxes have empty tensor."""
        dataset = CAMELDataset(
            root_dir=str(self.data_dir),
            split='train',
        )
        
        _, targets = dataset[0]
        
        self.assertEqual(targets['boxes'].shape[0], 0)
        self.assertEqual(targets['class_ids'].shape[0], 0)


class TestDataloaderCreation(unittest.TestCase):
    """Test dataloader creation helper function."""
    
    @classmethod
    def setUpClass(cls):
        """Create a temporary test dataset."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.data_dir = Path(cls.temp_dir) / "camel"
        
        # Create full structure
        for split in ['train', 'val']:
            (cls.data_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (cls.data_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        
        import cv2
        
        # Create test data for both splits
        for split in ['train', 'val']:
            img_dir = cls.data_dir / "images" / split
            lbl_dir = cls.data_dir / "labels" / split
            
            seq_name = "Seq01"
            for img_idx in range(4):
                img_path = img_dir / f"{seq_name}_{img_idx:03d}.png"
                img_data = np.random.randint(0, 255, (256, 336), dtype=np.uint8)
                cv2.imwrite(str(img_path), img_data)
            
            lbl_path = lbl_dir / f"{seq_name}.txt"
            with open(lbl_path, 'w') as f:
                f.write("0 0.5 0.5 0.3 0.4\n")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        shutil.rmtree(cls.temp_dir)
    
    def test_create_dataloaders(self):
        """Test dataloader creation."""
        train_loader, val_loader = create_dataloaders(
            data_dir=str(self.data_dir),
            batch_size=2,
            num_workers=0,
        )
        
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
    
    def test_dataloader_iteration(self):
        """Test that dataloaders can be iterated."""
        train_loader, val_loader = create_dataloaders(
            data_dir=str(self.data_dir),
            batch_size=2,
            num_workers=0,
        )
        
        # Get first batch from train loader
        for images, targets in train_loader:
            self.assertIsInstance(images, torch.Tensor)
            # Targets are dict with batched tensors
            self.assertIsInstance(targets, dict)
            self.assertIn('boxes', targets)
            self.assertIn('class_ids', targets)
            self.assertTrue(images.shape[0] > 0)
            break  # Just test first batch
    
    def test_batch_shape(self):
        """Test that batches have correct shape."""
        train_loader, _ = create_dataloaders(
            data_dir=str(self.data_dir),
            batch_size=2,
            num_workers=0,
        )
        
        for images, targets in train_loader:
            # Images should be (B, C, H, W)
            self.assertEqual(len(images.shape), 4)
            self.assertEqual(images.shape[0], 2)  # batch_size=2
            self.assertEqual(images.shape[1], 1)  # 1 channel
            self.assertEqual(images.shape[2], 256)  # height
            self.assertEqual(images.shape[3], 336)  # width
            
            # targets are dict with batched tensors
            self.assertIsInstance(targets, dict)
            self.assertIn('boxes', targets)
            self.assertIn('class_ids', targets)
            break


if __name__ == '__main__':
    unittest.main()
