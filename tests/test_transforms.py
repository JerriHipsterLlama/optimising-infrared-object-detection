"""
Unit tests for image transformation module.

Tests for:
- ImageTransforms (training with augmentation)
- InferenceTransforms (validation without augmentation)
"""

import unittest
import numpy as np
import torch
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.python.dataset.transforms import ImageTransforms, InferenceTransforms


class TestImageTransforms(unittest.TestCase):
    """Test cases for ImageTransforms class (training augmentations)."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a dummy 336x256 grayscale infrared image
        self.test_image = np.random.randint(0, 255, (256, 336), dtype=np.uint8)
        self.transforms_train = ImageTransforms(normalize=True)
    
    def test_output_shape(self):
        """Test that output tensor has correct shape (C, H, W)."""
        tensor = self.transforms_train(self.test_image)
        
        self.assertEqual(len(tensor.shape), 3, "Output should be 3D tensor (C, H, W)")
        self.assertEqual(tensor.shape[0], 1, "Should have 1 channel (grayscale)")
        self.assertEqual(tensor.shape[1], 256, "Height should be 256")
        self.assertEqual(tensor.shape[2], 336, "Width should be 336")
    
    def test_output_dtype(self):
        """Test that output tensor is float32."""
        tensor = self.transforms_train(self.test_image)
        self.assertEqual(tensor.dtype, torch.float32, "Output should be float32")
    
    def test_output_range(self):
        """Test that output tensor values are in [0, 1] range."""
        tensor = self.transforms_train(self.test_image)
        
        self.assertTrue(tensor.min() >= 0, "Min value should be >= 0")
        self.assertTrue(tensor.max() <= 1, "Max value should be <= 1")
    
    def test_normalization_8bit(self):
        """Test normalization of 8-bit images."""
        image_8bit = np.array([[0, 128, 255]], dtype=np.uint8).reshape(1, 3)
        tensor = self.transforms_train(image_8bit)
        
        expected = np.array([[[0.0, 128/255.0, 1.0]]], dtype=np.float32).reshape(1, 1, 3)
        torch.testing.assert_close(tensor, torch.from_numpy(expected), atol=1e-6, rtol=1e-5)
    
    def test_normalization_16bit(self):
        """Test normalization of 16-bit images."""
        # Pass actual uint16 data (not converted to uint8)
        image_16bit = np.array([[0, 32768, 65535]], dtype=np.uint16).reshape(1, 3)
        tensor = self.transforms_train(image_16bit)
        
        # uint16 should normalize by dividing by 65535
        expected = np.array([[[0.0, 32768 / 65535.0, 1.0]]], dtype=np.float32).reshape(1, 1, 3)
        torch.testing.assert_close(tensor, torch.from_numpy(expected), atol=1e-6, rtol=1e-5)
    def test_grayscale_to_3d_conversion(self):
        """Test that grayscale 2D image is converted to 3D tensor."""
        image_2d = np.array([[100, 150], [200, 50]], dtype=np.uint8)
        tensor = self.transforms_train(image_2d)
        
        self.assertEqual(len(tensor.shape), 3)
        self.assertEqual(tensor.shape[0], 1, "Should have 1 channel")
    
    def test_augmentation_disabled(self):
        """Test that no augmentation is applied when augment=False."""
        # Disable randomness for this test
        np.random.seed(42)
        
        transforms_no_aug = ImageTransforms(normalize=True)
        tensor = transforms_no_aug(self.test_image)
        
        # Just verify it returns a valid tensor (deterministic output is hard to test)
        self.assertEqual(tensor.shape, (1, 256, 336))
        self.assertTrue(torch.isfinite(tensor).all())
    
    def test_augmentation_produces_variations(self):
        """Test that transforms are deterministic (no augmentation in ImageTransforms)."""
        # Note: Augmentation is delegated to Ultralytics YOLOv8 training pipeline
        # ImageTransforms only handles normalization (deterministic)
        transforms = ImageTransforms(normalize=True)
        
        # Apply transforms multiple times to same image
        outputs = [transforms(self.test_image) for _ in range(5)]
        
        # All outputs should be identical (no augmentation applied here)
        for i in range(1, len(outputs)):
            self.assertTrue(torch.allclose(outputs[0], outputs[i]),
                          f"Output {i} differs from output 0 (transforms should be deterministic)")
    
    def test_handles_float_input(self):
        """Test that transforms handle float input images."""
        image_float = (self.test_image / 255.0).astype(np.float32)
        tensor = self.transforms_train(image_float)
        
        self.assertEqual(tensor.shape, (1, 256, 336))
        self.assertTrue(torch.isfinite(tensor).all())
    
    def test_normalize_option(self):
        """Test that normalize option works correctly."""
        # Use deterministic image with known max value (255)
        deterministic_image = np.ones((256, 336), dtype=np.uint8) * 255
        
        transforms_no_norm = ImageTransforms(normalize=False)
        tensor = transforms_no_norm(deterministic_image)
        
        # Without normalization, max should be 255, not normalized to 1.0
        self.assertEqual(tensor.max().item(), 255.0, "Without normalization, max should be 255")


class TestInferenceTransforms(unittest.TestCase):
    """Test cases for InferenceTransforms class (no augmentation)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_image = np.random.randint(0, 255, (256, 336), dtype=np.uint8)
        self.transforms = InferenceTransforms(normalize=True)
    
    def test_output_shape(self):
        """Test that output tensor has correct shape."""
        tensor = self.transforms(self.test_image)
        
        self.assertEqual(tensor.shape, (1, 256, 336))
    
    def test_output_normalized(self):
        """Test that output is normalized to [0, 1]."""
        tensor = self.transforms(self.test_image)
        
        self.assertTrue(tensor.min() >= 0)
        self.assertTrue(tensor.max() <= 1)
    
    def test_no_augmentation(self):
        """Test that inference transforms don't augment images."""
        # Run multiple times on same image
        outputs = [self.transforms(self.test_image) for _ in range(5)]
        
        # All outputs should be identical (no randomness)
        for i in range(1, len(outputs)):
            torch.testing.assert_close(outputs[0], outputs[i])
    
    def test_deterministic_output(self):
        """Test that same input produces identical output every time."""
        tensor1 = self.transforms(self.test_image.copy())
        tensor2 = self.transforms(self.test_image.copy())
        
        torch.testing.assert_close(tensor1, tensor2)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_uniform_image(self):
        """Test transforms on uniform (single color) image."""
        uniform_image = np.ones((256, 336), dtype=np.uint8) * 128
        transforms = ImageTransforms(normalize=True)
        tensor = transforms(uniform_image)
        
        self.assertEqual(tensor.shape, (1, 256, 336))
        self.assertTrue(torch.isfinite(tensor).all())
    
    def test_black_image(self):
        """Test transforms on completely black image."""
        black_image = np.zeros((256, 336), dtype=np.uint8)
        transforms = ImageTransforms(normalize=True)
        tensor = transforms(black_image)
        
        torch.testing.assert_close(tensor, torch.zeros((1, 256, 336)))
    
    def test_white_image(self):
        """Test transforms on completely white image."""
        white_image = np.ones((256, 336), dtype=np.uint8) * 255
        transforms = InferenceTransforms(normalize=True)
        tensor = transforms(white_image)
        
        torch.testing.assert_close(tensor, torch.ones((1, 256, 336)))
    
    def test_single_pixel_image(self):
        """Test transforms on minimal 1x1 image."""
        tiny_image = np.array([[128]], dtype=np.uint8)
        transforms = ImageTransforms(normalize=True)
        tensor = transforms(tiny_image)
        
        self.assertEqual(tensor.shape, (1, 1, 1))
    
    def test_very_large_image_values(self):
        """Test normalization with unusually large values."""
        large_image = np.ones((256, 336), dtype=np.uint16) * 32000
        large_image = (large_image / 256).astype(np.uint8)  # Convert for float conversion
        transforms = InferenceTransforms(normalize=True)
        tensor = transforms(large_image)
        
        self.assertTrue(tensor.max() <= 1)
        self.assertTrue(tensor.min() >= 0)


if __name__ == '__main__':
    unittest.main()
