"""
Plot Faster R-CNN Training Metrics

Visualizes training and validation metrics from results.csv and metrics.json files.
Creates plots similar to Ultralytics YOLO output, including results.png.

Usage:
    # From run directory
    python tools/plot_faster_rcnn_metrics.py --run models/checkpoints/fasterrcnn/train

    # From metrics file
    python tools/plot_faster_rcnn_metrics.py --metrics models/checkpoints/fasterrcnn/train/metrics.json

    # Save plots to custom location
    python tools/plot_faster_rcnn_metrics.py --run models/checkpoints/fasterrcnn/train --output plots/
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def plot_metrics_from_csv(csv_file: str, output_dir: str = None, show: bool = False):
    """
    Plot training metrics from results.csv file (YOLO-style format).

    Args:
        csv_file: Path to results.csv file
        output_dir: Directory to save plots (defaults to same directory as csv_file)
        show: Whether to display plots interactively
    """
    # Load CSV
    df = pd.read_csv(csv_file)

    # Determine output directory
    if output_dir is None:
        output_dir = Path(csv_file).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create main results plot (2x5 grid like YOLO)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Faster R-CNN Training Results', fontsize=16, fontweight='bold')

    # Flatten axes for easier indexing
    axes = axes.flatten()

    # Plot 1: Train Box Loss
    axes[0].plot(df['epoch'], df['train/box_loss'], color='#1f77b4', linewidth=2)
    axes[0].set_title('train/box_loss', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Train Classification Loss
    axes[1].plot(df['epoch'], df['train/cls_loss'], color='#ff7f0e', linewidth=2)
    axes[1].set_title('train/cls_loss', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('loss')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Train Objectness Loss
    axes[2].plot(df['epoch'], df['train/objectness_loss'], color='#2ca02c', linewidth=2)
    axes[2].set_title('train/objectness_loss', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('epoch')
    axes[2].set_ylabel('loss')
    axes[2].grid(True, alpha=0.3)

    # Plot 4: Train RPN Box Loss
    axes[3].plot(df['epoch'], df['train/rpn_box_loss'], color='#d62728', linewidth=2)
    axes[3].set_title('train/rpn_box_loss', fontsize=11, fontweight='bold')
    axes[3].set_xlabel('epoch')
    axes[3].set_ylabel('loss')
    axes[3].grid(True, alpha=0.3)

    # Plot 5: Learning Rate
    axes[4].plot(df['epoch'], df['lr'], color='#9467bd', linewidth=2)
    axes[4].set_title('lr', fontsize=11, fontweight='bold')
    axes[4].set_xlabel('epoch')
    axes[4].set_ylabel('learning rate')
    axes[4].grid(True, alpha=0.3)

    # Plot 6: Val Box Loss
    val_data = df[df['val/box_loss'] > 0]
    if len(val_data) > 0:
        axes[5].plot(val_data['epoch'], val_data['val/box_loss'], color='#1f77b4', linewidth=2, marker='o')
        axes[5].set_title('val/box_loss', fontsize=11, fontweight='bold')
        axes[5].set_xlabel('epoch')
        axes[5].set_ylabel('loss')
        axes[5].grid(True, alpha=0.3)

    # Plot 7: Val Classification Loss
    if len(val_data) > 0:
        axes[6].plot(val_data['epoch'], val_data['val/cls_loss'], color='#ff7f0e', linewidth=2, marker='o')
        axes[6].set_title('val/cls_loss', fontsize=11, fontweight='bold')
        axes[6].set_xlabel('epoch')
        axes[6].set_ylabel('loss')
        axes[6].grid(True, alpha=0.3)

    # Plot 8: Precision & Recall
    if len(val_data) > 0:
        axes[7].plot(val_data['epoch'], val_data['metrics/precision'], label='Precision', color='#1f77b4', linewidth=2, marker='o')
        axes[7].plot(val_data['epoch'], val_data['metrics/recall'], label='Recall', color='#ff7f0e', linewidth=2, marker='s')
        axes[7].set_title('metrics/precision & recall', fontsize=11, fontweight='bold')
        axes[7].set_xlabel('epoch')
        axes[7].set_ylabel('value')
        axes[7].legend()
        axes[7].grid(True, alpha=0.3)
        axes[7].set_ylim([0, 1])

    # Plot 9: mAP50 & mAP50-95
    if len(val_data) > 0:
        axes[8].plot(val_data['epoch'], val_data['metrics/mAP50'], label='mAP@0.5', color='#2ca02c', linewidth=2, marker='o')
        axes[8].plot(val_data['epoch'], val_data['metrics/mAP50-95'], label='mAP@0.5:0.95', color='#d62728', linewidth=2, marker='s')
        axes[8].set_title('metrics/mAP', fontsize=11, fontweight='bold')
        axes[8].set_xlabel('epoch')
        axes[8].set_ylabel('mAP')
        axes[8].legend()
        axes[8].grid(True, alpha=0.3)
        axes[8].set_ylim([0, 1])

    # Plot 10: Empty (reserved for future use)
    axes[9].axis('off')

    plt.tight_layout()

    # Save as results.png (YOLO-style naming)
    save_path = output_dir / "results.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Results plot saved to: {save_path}")

    if show:
        plt.show()

    plt.close()


def print_summary(metrics_file: str):
    """Print training summary statistics."""
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    print("\n" + "=" * 80)
    print("Faster R-CNN Training Summary")
    print("=" * 80)
    print(f"Total Epochs: {len(metrics['train_loss'])}")
    print(f"Final Train Loss: {metrics['train_loss'][-1]:.4f}")

    if metrics['val_loss']:
        # Find last non-NaN validation loss
        valid_val_losses = [x for x in metrics['val_loss'] if not np.isnan(x)]
        if valid_val_losses:
            print(f"Best Val Loss: {min(valid_val_losses):.4f}")
            print(f"Final Val Loss: {valid_val_losses[-1]:.4f}")

    if metrics.get('val_mAP50'):
        valid_map50 = [x for x in metrics['val_mAP50'] if not np.isnan(x)]
        if valid_map50:
            print(f"Best mAP@0.5: {max(valid_map50):.4f}")
            print(f"Final mAP@0.5: {valid_map50[-1]:.4f}")

    if metrics.get('val_precision'):
        valid_precision = [x for x in metrics['val_precision'] if not np.isnan(x)]
        if valid_precision:
            print(f"Final Precision: {valid_precision[-1]:.4f}")

    if metrics.get('val_recall'):
        valid_recall = [x for x in metrics['val_recall'] if not np.isnan(x)]
        if valid_recall:
            print(f"Final Recall: {valid_recall[-1]:.4f}")

    print("=" * 80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Faster R-CNN training metrics')
    parser.add_argument('--run', type=str, default=None, help='Path to run directory (e.g., models/checkpoints/fasterrcnn/train)')
    parser.add_argument('--metrics', type=str, default=None, help='Path to metrics.json file (legacy support)')
    parser.add_argument('--output', type=str, default=None, help='Directory to save plots (optional)')
    parser.add_argument('--show', action='store_true', help='Display plots interactively')
    parser.add_argument('--summary', action='store_true', help='Print summary statistics')

    args = parser.parse_args()

    # Determine which file to use
    if args.run:
        run_dir = Path(args.run)
        csv_file = run_dir / "results.csv"
        metrics_file = run_dir / "metrics.json"

        if not csv_file.exists():
            print(f"Error: results.csv not found in {run_dir}")
            exit(1)

        # Print summary if requested
        if args.summary and metrics_file.exists():
            print_summary(str(metrics_file))

        # Plot from CSV (primary method)
        print(f"Plotting metrics from: {csv_file}")
        plot_metrics_from_csv(str(csv_file), output_dir=args.output, show=args.show)

    elif args.metrics:
        # Legacy support for direct metrics.json path
        metrics_file = Path(args.metrics)
        csv_file = metrics_file.parent / "results.csv"

        if not metrics_file.exists():
            print(f"Error: Metrics file not found: {metrics_file}")
            exit(1)

        # Print summary if requested
        if args.summary:
            print_summary(str(metrics_file))

        # Prefer CSV if available, otherwise use metrics.json
        if csv_file.exists():
            print(f"Plotting metrics from: {csv_file}")
            plot_metrics_from_csv(str(csv_file), output_dir=args.output, show=args.show)
        else:
            print("Warning: results.csv not found, plotting functionality limited")
            print("Please use the new training script that generates results.csv")

    else:
        print("Error: Must specify either --run or --metrics")
        parser.print_help()
        exit(1)
