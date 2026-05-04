"""Quick script to check for empty label files in validation set."""
from pathlib import Path

labels_dir = Path("data/camel/labels_pascal/val")
empty_files = []

for label_file in sorted(labels_dir.glob("*.txt")):
    content = label_file.read_text().strip()
    if not content:
        empty_files.append(label_file.name)

print(f"Total label files checked: {len(list(labels_dir.glob('*.txt')))}")
print(f"Empty label files found: {len(empty_files)}")
print(f"\nFirst 20 empty label files:")
for fname in empty_files[:20]:
    print(f"  {fname}")
