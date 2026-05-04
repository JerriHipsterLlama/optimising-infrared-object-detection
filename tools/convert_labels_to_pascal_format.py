"""
Convert CAMEL dataset labels from per-sequence format to per-image Pascal VOC format.

Original structure:
- Labels in data/camel/labels_original/train/Seq##.txt (one file per sequence)
- Format: Frame Number, Track ID, Class ID, x_topleft, y_topleft, width, height
- Multiple annotations per frame are stored as separate lines

Pascal VOC format for Faster R-CNN:
- Labels in data/camel/labels_pascal/train/Seq##_XXXXXX.txt (one file per frame/image)
- Multiple bounding boxes in same frame go in ONE file (one bbox per line)
- Format: class_id (0-indexed), x1, y1, x2, y2 (corner coordinates in pixels)
- Empty files for frames with no objects (background class handled by Faster R-CNN)

Class Mapping (0-indexed):
- 1 -> 0 (person)
- 2 -> 1 (bicycle)
- 3 -> 2 (vehicle)
- 18 -> 3 (dog)

Note: Training script will add 1 to class IDs for Faster R-CNN (0 = background)

Usage:
    python tools/convert_labels_to_pascal_format.py
    python tools/convert_labels_to_pascal_format.py --split all
    python tools/convert_labels_to_pascal_format.py --split val
"""

import os
from pathlib import Path
from collections import defaultdict
import argparse


# CAMEL class mapping: original_class_id -> pascal_class_id
CLASS_MAPPING = {
    1: 0,    # person
    2: 1,    # bicycle
    3: 2,    # vehicle
    18: 3,   # dog
}


def remap_class_id(original_class_id: int) -> int:
    """
    Remap original CAMEL class ID to Pascal format (0-indexed).
    
    Args:
        original_class_id (int): Original class ID from CAMEL dataset
    
    Returns:
        int: Remapped class ID (0-indexed), or original if not in mapping
    """
    return CLASS_MAPPING.get(original_class_id, original_class_id)


def convert_sequence_labels_to_pascal(labels_dir: str, split: str = "train"):
    """
    Convert CAMEL per-sequence label files to per-image Pascal VOC format label files.
    
    Reads from: data/camel/labels_original/{split}/Seq##.txt
    Writes to: data/camel/labels_pascal/{split}/Seq##_XXXXXX.txt
    
    CAMEL format (per annotation line):
        Frame Number, Track ID, Class ID, x_topleft, y_topleft, width, height
    
    Pascal VOC format (per bounding box):
        Class ID (0-indexed), x1, y1, x2, y2 (corner coordinates in pixels)
    
    Args:
        labels_dir (str): Path to data/camel directory
        split (str): Dataset split ('train', 'val', 'test'). Default: 'train'
    """
    labels_original_path = Path(labels_dir) / "labels_original" / split
    labels_output_path = Path(labels_dir) / "labels_pascal" / split
    images_path = Path(labels_dir) / "images" / split
    
    # CAMEL infrared image dimensions
    IMG_WIDTH = 336
    IMG_HEIGHT = 256
    
    if not labels_original_path.exists():
        print(f"❌ Labels directory not found: {labels_original_path}")
        return False
    
    if not images_path.exists():
        print(f"❌ Images directory not found: {images_path}")
        return False
    
    # Create output labels directory if it doesn't exist
    labels_output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Converting labels for {split} split to Pascal VOC format")
    print(f"{'='*80}")
    print(f"Reading from: {labels_original_path}")
    print(f"Writing to:   {labels_output_path}")
    print(f"Image dimensions: {IMG_WIDTH}x{IMG_HEIGHT}")
    print(f"Format: class_id x1 y1 x2 y2 (corner coordinates in pixels)")
    print()
    
    # Get all sequence label files from labels_original
    seq_label_files = sorted(labels_original_path.glob("Seq*.txt"))
    
    if not seq_label_files:
        print(f"❌ No sequence label files found in {labels_original_path}")
        return False
    
    print(f"Found {len(seq_label_files)} sequence label files")
    
    # Group labels by frame (image)
    image_labels = defaultdict(list)
    
    for label_file in seq_label_files:
        seq_name = label_file.stem  # e.g., "Seq01"
        
        print(f"  Processing {seq_name}...")
        
        # Read all annotations from the sequence file
        annotations = []
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                annotations.append(line)
        
        print(f"    Read {len(annotations)} annotations")
        
        # Parse annotations and group by frame
        for annotation in annotations:
            parts = annotation.split()
            
            if len(parts) < 7:
                print(f"    WARNING: Invalid annotation (expected 7+ fields): {annotation}")
                continue
            
            try:
                frame_num = int(parts[0])
                track_id = int(parts[1])
                class_id = int(parts[2])
                x_topleft = float(parts[3])
                y_topleft = float(parts[4])
                width = float(parts[5])
                height = float(parts[6])
                
                # Remap class ID to 0-indexed Pascal format
                class_id_remapped = remap_class_id(class_id)
                
                # Convert to corner coordinates (x1, y1, x2, y2) in pixels
                x1 = x_topleft
                y1 = y_topleft
                x2 = x_topleft + width
                y2 = y_topleft + height
                
                # Clamp to image boundaries
                x1 = max(0.0, min(IMG_WIDTH, x1))
                y1 = max(0.0, min(IMG_HEIGHT, y1))
                x2 = max(0.0, min(IMG_WIDTH, x2))
                y2 = max(0.0, min(IMG_HEIGHT, y2))
                
                # Validate box (ensure x2 > x1 and y2 > y1)
                if x2 <= x1 or y2 <= y1:
                    print(f"    WARNING: Invalid box dimensions: ({x1}, {y1}, {x2}, {y2})")
                    continue
                
                # Create image name: Seq##_XXXXXX (frame number is 1-indexed in annotations)
                img_name = f"{seq_name}_{frame_num:06d}"
                
                # Format as Pascal VOC: class_id x1 y1 x2 y2
                pascal_label = f"{class_id_remapped} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}"
                image_labels[img_name].append(pascal_label)
                
            except (ValueError, IndexError) as e:
                print(f"    WARNING: Failed to parse annotation: {annotation} ({e})")
                continue
    
    # Write individual label files for each image
    # All annotations for the same frame go into ONE label file
    created_count = 0
    
    for img_name, labels in image_labels.items():
        label_file = labels_output_path / f"{img_name}.txt"
        
        # Write label file (one bbox per line, multiple bboxes if multiple objects in frame)
        with open(label_file, 'w') as f:
            for label in labels:
                f.write(label + '\n')
        
        created_count += 1
        if created_count % 1000 == 0:
            print(f"    Created {created_count} label files...")
    
    # Create empty label files for images without annotations
    # Faster R-CNN handles empty files as background class (no objects)
    all_images = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.npy']:
        all_images.extend(sorted(images_path.glob(f"Seq*{ext}")))
    all_images = sorted(set(all_images))
    
    print(f"\n  Creating empty label files for images without annotations...")
    empty_count = 0
    for img_path in all_images:
        img_name = img_path.stem
        label_file = labels_output_path / f"{img_name}.txt"
        
        if not label_file.exists():
            # Create empty label file for images with no annotations
            # Faster R-CNN will treat these as background class (no objects)
            label_file.touch()
            empty_count += 1
    
    print(f"\n{'='*80}")
    print(f"✓ Conversion complete!")
    print(f"{'='*80}")
    print(f"  Created {created_count} label files with annotations")
    print(f"  Created {empty_count} empty label files (images with no annotations)")
    print(f"  Total: {created_count + empty_count} label files")
    print()
    
    # Show sample of converted labels
    if created_count > 0:
        sample_labels = list(image_labels.items())[:3]
        print(f"Sample converted labels (first 3 images with annotations):")
        for img_name, labels in sample_labels:
            print(f"\n  {img_name}.txt:")
            for label in labels[:5]:  # Show up to 5 boxes per image
                parts = label.split()
                class_id = int(parts[0])
                class_name = {0: 'person', 1: 'bicycle', 2: 'vehicle', 3: 'dog'}.get(class_id, 'unknown')
                print(f"    {label}  # {class_name}")
            if len(labels) > 5:
                print(f"    ... and {len(labels) - 5} more boxes")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert CAMEL labels from per-sequence to per-image Pascal VOC format for Faster R-CNN',
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/camel',
        help='Path to data/camel directory (default: data/camel)',
    )
    
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'test', 'all'],
        help='Dataset split to convert (default: train)',
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"CAMEL to Pascal VOC Format Converter for Faster R-CNN")
    print(f"{'='*80}")
    print(f"Data directory: {args.data_dir}")
    print(f"Split: {args.split}")
    
    if args.split == 'all':
        success_count = 0
        for split in ['train', 'val', 'test']:
            if convert_sequence_labels_to_pascal(args.data_dir, split):
                success_count += 1
        
        print(f"\n{'='*80}")
        print(f"✓ Completed {success_count}/3 splits successfully")
        print(f"{'='*80}")
    else:
        convert_sequence_labels_to_pascal(args.data_dir, args.split)


if __name__ == '__main__':
    main()
