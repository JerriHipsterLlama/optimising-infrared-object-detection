#!/usr/bin/env python3
"""Convert Seq*.txt annotation files (frame, track, class, top-left-x, top-left-y, w, h)
into per-image YOLO-format label files (class x_center_norm y_center_norm w_norm h_norm).

Usage:
  python scripts/seq_to_yolo.py --seq-dir data --images-dir data/camel/images --out-label-dir data/camel/labels --dry-run
"""
import argparse
from pathlib import Path
import re
from PIL import Image
import sys


SEQ_RE = re.compile(r"(Seq\d+)")
IMG_NAME_RE = re.compile(r"(Seq\d+)_(\d+)\.[jJ][pP][gG]$")


def build_image_index(images_dir: Path):
    idx = {}
    for p in images_dir.rglob("*.jpg"):
        m = IMG_NAME_RE.match(p.name)
        if not m:
            continue
        seq = m.group(1)
        frame = int(m.group(2))
        idx.setdefault(seq, {})[frame] = p
    return idx


def convert_seq_file(seq_file: Path, image_index, out_label_dir: Path, dry_run=False):
    seq_name_m = SEQ_RE.search(seq_file.stem)
    if not seq_name_m:
        print(f"Skipping {seq_file} (no seq name)")
        return 0
    seq_name = seq_name_m.group(1)
    print(f"Processing {seq_name} from {seq_file}")

    images_for_seq = image_index.get(seq_name, {})
    if not images_for_seq:
        print(f"  No images found for {seq_name}, skipping")
        return 0

    written = 0
    with open(seq_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            frame = int(parts[0])
            cls = int(parts[2])
            x = float(parts[3])
            y = float(parts[4])
            w = float(parts[5])
            h = float(parts[6])

            img_path = images_for_seq.get(frame)
            if img_path is None:
                # try zero-padded or other formats
                img_path = images_for_seq.get(int(str(frame)), None)
            if img_path is None:
                # log and skip
                # print(f"  Missing image for frame {frame} in {seq_name}")
                continue

            # read image size
            with Image.open(img_path) as im:
                iw, ih = im.size

            # convert top-left (x,y,w,h) to center and normalize
            x_c = x + w / 2.0
            y_c = y + h / 2.0
            x_n = max(0.0, min(1.0, x_c / iw))
            y_n = max(0.0, min(1.0, y_c / ih))
            w_n = max(0.0, min(1.0, w / iw))
            h_n = max(0.0, min(1.0, h / ih))

            out_label_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_label_dir / (img_path.stem + ".txt")
            line_out = f"{cls} {x_n:.6f} {y_n:.6f} {w_n:.6f} {h_n:.6f}\n"
            if not dry_run:
                with open(out_file, "a") as of:
                    of.write(line_out)
            written += 1

    print(f"  Wrote/queued {written} labels for {seq_name}")
    return written


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-dir", default="data", help="Directory containing Seq*-IR.txt files")
    p.add_argument("--images-dir", default="data/camel/images", help="Root images directory (will search recursively)")
    p.add_argument("--out-label-dir", default="data/camel/labels", help="Output labels directory")
    p.add_argument("--dry-run", action="store_true", help="Don't write files, just report")
    args = p.parse_args()

    seq_dir = Path(args.seq_dir)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_label_dir)

    print(f"Indexing images under {images_dir}")
    image_index = build_image_index(images_dir)
    if not image_index:
        print("No images found; check --images-dir")
        sys.exit(2)

    # deduplicate matches (Seq*-IR.txt may also match Seq*.txt)
    seq_files = sorted(list({p for p in seq_dir.glob("Seq*-IR.txt")} | {p for p in seq_dir.glob("Seq*.txt")}))
    total = 0
    for sf in seq_files:
        try:
            total += convert_seq_file(sf, image_index, out_dir, dry_run=args.dry_run)
        except FileNotFoundError:
            print(f"Warning: seq file disappeared while processing: {sf}")
            continue

    print(f"Conversion complete: total labels processed {total}")


if __name__ == "__main__":
    main()
