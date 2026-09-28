import argparse
from pathlib import Path
import shutil


def has_drone(label_path: Path) -> bool:
    """Check if label file contains class 0 (drone)."""
    if not label_path.exists():
        return False
    with open(label_path) as f:
        for line in f:
            if line.strip().startswith("0 "):
                return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)

    for class_name in ["drone", "background"]:
        (output_dir / class_name).mkdir(parents=True, exist_ok=True)

    counts = {"drone": 0, "background": 0}

    for img_path in images_dir.glob("*.png"):
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        if has_drone(label_path):
            dest = output_dir / "drone" / img_path.name
            counts["drone"] += 1
        else:
            dest = output_dir / "background" / img_path.name
            counts["background"] += 1
        
        shutil.copy(img_path, dest)

    print(f"Organized: drone={counts['drone']}, background={counts['background']}")


if __name__ == "__main__":
    main()
