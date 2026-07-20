"""
Convert CAMEL dataset labels from per-sequence format to per-image YOLO format.

Original structure:
- Labels in data/camel/labels/train/Seq##.txt (one file per sequence)
- Format: Frame Number, Track ID, Class ID, x_topleft, y_topleft, width, height
- Multiple annotations per frame are stored as separate lines

Expected YOLO format:
- Labels in data/camel/labels/train/Seq##_XXXXXX.txt (one file per frame/image)
- Multiple bounding boxes in same frame go in ONE file (one bbox per line)
- Format: class_id (0-indexed), x_center (normalized), y_center (normalized), width (normalized), height (normalized)

Class Mapping (0-indexed):
- 1 -> 0 (person)
- 2 -> 1 (bicycle)
- 3 -> 2 (vehicle)
- 18 -> 3 (dog)

Usage:
    python scripts/convert_labels_to_yolo_format.py
"""

import os
from pathlib import Path
from collections import defaultdict
import argparse


# CAMEL class mapping: original_class_id -> yolo_class_id
CLASS_MAPPING = {
    1: 0,    # person
    2: 1,    # bicycle
    3: 2,    # vehicle
    18: 3,   # dog
}


def remap_class_id(original_class_id: int) -> int:
    """
    Remap original CAMEL class ID to YOLO format (0-indexed).
    
    Args:
        original_class_id (int): Original class ID from CAMEL dataset
    
    Returns:
        int: Remapped class ID (0-indexed), or original if not in mapping
    """
    return CLASS_MAPPING.get(original_class_id, original_class_id)



def convert_sequence_labels_to_image_labels(labels_dir: str, split: str = "train"):
    """
    Convert CAMEL per-sequence label files to per-image YOLO format label files.
    
    Reads from: data/camel/labels_original/{split}/Seq##.txt
    Writes to: data/camel/labels/{split}/Seq##_XXXXXX.txt
    
    CAMEL format (per annotation line):
        Frame Number, Track ID, Class ID, x_topleft, y_topleft, width, height
    
    YOLO format (per bounding box):
        Class ID, x_center (normalized), y_center (normalized), width (normalized), height (normalized)
    
    Args:
        labels_dir (str): Path to data/camel directory
        split (str): Dataset split ('train', 'val', 'test'). Default: 'train'
    """
    labels_original_path = Path(labels_dir) / "labels_original" / split
    labels_output_path = Path(labels_dir) / "labels" / split
    images_path = Path(labels_dir) / "images" / split
    
    # CAMEL infrared image dimensions
    IMG_WIDTH = 336
    IMG_HEIGHT = 256
    
    if not labels_original_path.exists():
        print(f"Labels directory not found: {labels_original_path}")
        return False
    
    if not images_path.exists():
        print(f"Images directory not found: {images_path}")
        return False
    
    # Create output labels directory if it doesn't exist
    labels_output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting labels for {split} split...")
    print(f"Reading from: {labels_original_path}")
    print(f"Writing to: {labels_output_path}")
    print(f"Image dimensions: {IMG_WIDTH}x{IMG_HEIGHT}")
    
    # Get all sequence label files from labels_original
    seq_label_files = sorted(labels_original_path.glob("Seq*.txt"))
    
    if not seq_label_files:
        print(f"No sequence label files found in {labels_original_path}")
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
                
                # Remap class ID to 0-indexed YOLO format
                class_id_remapped = remap_class_id(class_id)
                
                # Convert from top-left + size to center + normalized size
                x_center = (x_topleft + width / 2) / IMG_WIDTH
                y_center = (y_topleft + height / 2) / IMG_HEIGHT
                width_norm = width / IMG_WIDTH
                height_norm = height / IMG_HEIGHT
                
                # Clamp to [0, 1]
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                width_norm = max(0.0, min(1.0, width_norm))
                height_norm = max(0.0, min(1.0, height_norm))
                
                # Create image name: Seq##_XXXXXX (frame number is 1-indexed in annotations)
                img_name = f"{seq_name}_{frame_num:06d}"
                
                # Format as YOLO: class_id x_center y_center width height
                yolo_label = f"{class_id_remapped} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"
                image_labels[img_name].append(yolo_label)
                
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
    all_images = sorted(images_path.glob("Seq*.png"))
    all_images.extend(sorted(images_path.glob("Seq*.jpg")))
    all_images.extend(sorted(images_path.glob("Seq*.jpeg")))
    all_images = sorted(set(all_images))
    
    empty_count = 0
    for img_path in all_images:
        img_name = img_path.stem
        label_file = labels_output_path / f"{img_name}.txt"
        
        if not label_file.exists():
            # Create empty label file for images with no annotations
            label_file.touch()
            empty_count += 1
    
    print(f"\n✓ Conversion complete!")
    print(f"  Created {created_count} label files with annotations")
    print(f"  Created {empty_count} empty label files (images with no annotations)")
    print(f"  Total: {created_count + empty_count} label files")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert CAMEL labels from per-sequence to per-image format for YOLO',
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
    
    if args.split == 'all':
        for split in ['train', 'val', 'test']:
            convert_sequence_labels_to_image_labels(args.data_dir, split)
            print()
    else:
        convert_sequence_labels_to_image_labels(args.data_dir, args.split)


if __name__ == '__main__':
    main()
