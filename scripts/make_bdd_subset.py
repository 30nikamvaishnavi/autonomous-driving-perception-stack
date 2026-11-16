"""Create a reproducible BDD100K detection subset for Kaggle training."""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

SPLIT_SIZES = {"train": 10_000, "val": 2_000}
CATEGORY_ALIASES = {"bike": "bicycle", "motor": "motorcycle"}


def build_split(
    source: Path,
    destination: Path,
    split: str,
    count: int,
    seed: int = 42,
) -> int:
    """Copy ``count`` image/label records into a split using a seeded sample."""
    labels_path = source / "labels" / "det_20" / f"det_{split}.json"
    images_dir = source / "images" / split
    with labels_path.open(encoding="utf-8") as file:
        frames = json.load(file)

    available = [frame for frame in frames if (images_dir / frame["name"]).is_file()]
    if len(available) < count:
        raise ValueError(f"{split} has {len(available)} available images; {count} required")

    selected = random.Random(seed).sample(available, count)
    selected = [json.loads(json.dumps(frame)) for frame in selected]
    target_images = destination / "images" / split
    target_labels = destination / "labels" / "det_20"
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

    for frame in selected:
        for label in frame.get("labels") or []:
            category = label.get("category")
            label["category"] = CATEGORY_ALIASES.get(category, category)
        shutil.copy2(images_dir / frame["name"], target_images / frame["name"])

    output_labels = target_labels / f"det_{split}.json"
    output_labels.write_text(json.dumps(selected, ensure_ascii=False), encoding="utf-8")
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="BDD100K root")
    parser.add_argument("--out", type=Path, required=True, help="Output subset root")
    args = parser.parse_args()

    for split, count in SPLIT_SIZES.items():
        written = build_split(args.source, args.out, split, count)
        print(f"{split}: {written} images")


if __name__ == "__main__":
    main()
