"""
Organize CAMEL dataset images and labels into train/val/test splits.

This script copies images and labels from a staging directory into the 
proper train/val/test splits for training.

Dataset organization strategy:
- Train: Seq01, Seq05-Seq17, Seq19-Seq20, Seq23, Seq25-Seq30, Seq02, Seq03, Seq04
- Val: Seq21
- Test: Seq15, Seq18

Usage:
    python scripts/organize_dataset.py --source data/camel_images --dest data/camel
"""

import shutil
import argparse
from pathlib import Path
from collections import defaultdict


# Define which sequences go to which splits
SPLIT_MAPPING = {
    'train': ['Seq01', 'Seq02', 'Seq03', 'Seq04', 'Seq05', 'Seq07', 'Seq08', 
              'Seq10', 'Seq11', 'Seq13', 'Seq17', 'Seq19', 'Seq25', 'Seq26', 'Seq27', 'Seq28', 'Seq29', 'Seq30'],
    'val': ['Seq06', 'Seq09', 'Seq20', 'Seq23'],
    'test': ['Seq15', 'Seq18', 'Seq21'],
}


def organize_images(source_dir: str, dest_dir: str):
    """
    Organize images into train/val/test splits.
    
    Args:
        source_dir (str): Path to source directory containing Seq## or IR ## folders
        dest_dir (str): Path to data/camel directory
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    if not source_path.exists():
        print(f"Source directory not found: {source_path}")
        return
    
    print(f"Organizing images from {source_path} to {dest_path}")
    
    # Create destination directories if they don't exist
    for split in ['train', 'val', 'test']:
        split_dir = dest_path / 'images' / split
        split_dir.mkdir(parents=True, exist_ok=True)
    
    # Mapping from IR ## to Seq##
    ir_to_seq = {
        'IR 1': 'Seq01', 'IR 2': 'Seq02', 'IR 3': 'Seq03', 'IR 4': 'Seq04',
        'IR 5': 'Seq05', 'IR 6': 'Seq06', 'IR 7': 'Seq07', 'IR 8': 'Seq08',
        'IR 9': 'Seq09', 'IR 10': 'Seq10', 'IR 11': 'Seq11', 'IR 13': 'Seq13',
        'IR 15': 'Seq15', 'IR 17': 'Seq17', 'IR 18': 'Seq18', 'IR 19': 'Seq19',
        'IR 20': 'Seq20', 'IR 21': 'Seq21', 'IR 23': 'Seq23', 'IR 25': 'Seq25',
        'IR 26': 'Seq26', 'IR 27': 'Seq27', 'IR 28': 'Seq28', 'IR 29': 'Seq29',
        'IR 30': 'Seq30',
    }
    
    # Process each sequence
    for split, sequences in SPLIT_MAPPING.items():
        for seq_name in sequences:
            # Find sequence folder (might be "IR ##" or "Seq##")
            seq_num = seq_name.replace('Seq', '')
            
            # Try both naming conventions
            seq_folders = list(source_path.glob(f"*{seq_num}*"))
            if not seq_folders:
                # Try IR naming
                ir_name = f"IR {int(seq_num)}"
                seq_folders = list(source_path.glob(ir_name))
            
            if not seq_folders:
                print(f"  WARNING: No folder found for {seq_name} (tried Seq{seq_num} and IR {int(seq_num)})")
                continue
            
            seq_dir = seq_folders[0]
            print(f"  Copying {seq_name} images to {split} (from {seq_dir.name})...")
            
            # Copy all images from sequence folder
            image_files = list(seq_dir.glob('*.png')) + list(seq_dir.glob('*.jpg'))
            
            if not image_files:
                print(f"    WARNING: No images found in {seq_dir}")
                continue
            
            for img_file in image_files:
                dest_file = dest_path / 'images' / split / img_file.name
                
                # Rename if needed to match expected format (Seq##_XXXXXX.png)
                if not img_file.name.startswith(seq_name):
                    # Extract frame number and rename
                    frame_num = img_file.stem.split('_')[-1] if '_' in img_file.stem else img_file.stem
                    new_name = f"{seq_name}_{frame_num}.png"
                    dest_file = dest_path / 'images' / split / new_name
                
                shutil.copy2(img_file, dest_file)
            
            print(f"    ✓ Copied {len(image_files)} images")
    
    print(f"\n✓ Image organization complete!")


def organize_labels(source_dir: str, dest_dir: str):
    """
    Organize label files into train/val/test splits.
    
    Args:
        source_dir (str): Path to source directory containing Seq##.txt files
        dest_dir (str): Path to data/camel directory
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    if not source_path.exists():
        print(f"Source directory not found: {source_path}")
        return
    
    print(f"Organizing labels from {source_path} to {dest_path}")
    
    # Create destination directories if they don't exist
    for split in ['train', 'val', 'test']:
        split_dir = dest_path / 'labels' / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Also create labels_original for backup
        orig_dir = dest_path / 'labels_original' / split
        orig_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each sequence
    for split, sequences in SPLIT_MAPPING.items():
        for seq_name in sequences:
            label_file = source_path / f"{seq_name}.txt"
            
            if not label_file.exists():
                print(f"  WARNING: Label file not found: {label_file}")
                continue
            
            print(f"  Copying {seq_name}.txt to {split}...")
            
            # Copy to labels_original (keep original format)
            orig_dest = dest_path / 'labels_original' / split / label_file.name
            shutil.copy2(label_file, orig_dest)
            
            # Copy to labels (will be converted by convert_labels_to_yolo_format.py)
            labels_dest = dest_path / 'labels' / split / label_file.name
            shutil.copy2(label_file, labels_dest)
            
            print(f"    ✓ Copied {label_file.name}")
    
    print(f"\n✓ Label organization complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Organize CAMEL dataset into train/val/test splits',
    )
    
    parser.add_argument(
        '--source-images',
        type=str,
        default='data/camel_images',
        help='Path to source image directory (default: data/camel_images)',
    )
    
    parser.add_argument(
        '--source-labels',
        type=str,
        default='data',
        help='Path to source label directory (default: data)',
    )
    
    parser.add_argument(
        '--dest',
        type=str,
        default='data/camel',
        help='Path to destination data/camel directory (default: data/camel)',
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CAMEL Dataset Organization")
    print("=" * 60)
    print()
    print("Split mapping:")
    for split, sequences in SPLIT_MAPPING.items():
        print(f"  {split:5s}: {', '.join(sequences)}")
    print()
    
    organize_images(args.source_images, args.dest)
    print()
    organize_labels(args.source_labels, args.dest)
    
    print()
    print("=" * 60)
    print("Next steps:")
    print("  1. Verify images are organized correctly")
    print("  2. Run: python scripts/convert_labels_to_yolo_format.py --split all")
    print("=" * 60)


if __name__ == '__main__':
    main()
