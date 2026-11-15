"""Convert BDD100K detection labels (JSON) to YOLO txt format + dataset.yaml."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from adpstack.utils.logging import get_logger

log = get_logger(__name__)

CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle", "person", "rider",
           "traffic light", "traffic sign"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
CATEGORY_ALIASES = {"bike": "bicycle", "motor": "motorcycle"}
IMG_W, IMG_H = 1280, 720


def box_to_yolo(x1: float, y1: float, x2: float, y2: float,
                w: int = IMG_W, h: int = IMG_H) -> tuple[float, float, float, float] | None:
    """Pixel xyxy -> normalized (cx, cy, bw, bh), clipped to the image. None if degenerate."""
    x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
    y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
    bw, bh = x2 - x1, y2 - y1
    if bw < 1.0 or bh < 1.0:
        return None
    return ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, bw / w, bh / h)


def frame_to_yolo_lines(frame: dict, w: int = IMG_W, h: int = IMG_H) -> list[str]:
    """Convert one BDD100K frame record to YOLO label lines."""
    lines = []
    for lab in frame.get("labels") or []:
        cat = lab.get("category")
        cat = CATEGORY_ALIASES.get(cat, cat)
        box = lab.get("box2d")
        if cat not in CLASS_TO_ID or not box:
            continue
        y = box_to_yolo(box["x1"], box["y1"], box["x2"], box["y2"], w, h)
        if y is None:
            continue
        lines.append(f"{CLASS_TO_ID[cat]} {y[0]:.6f} {y[1]:.6f} {y[2]:.6f} {y[3]:.6f}")
    return lines


def convert_split(labels_json: Path, images_dir: Path, out_dir: Path, split: str,
                  subset: int | None = None, copy_images: bool = False) -> int:
    """Convert one split. Images are symlinked (or copied) next to the labels."""
    with open(labels_json, encoding="utf-8") as f:
        frames = json.load(f)
    if subset:
        frames = frames[:subset]
    (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    n = 0
    for fr in frames:
        stem = Path(fr["name"]).stem
        src = images_dir / fr["name"]
        if not src.exists():
            continue
        dst = out_dir / "images" / split / src.name
        if not dst.exists():
            if copy_images:
                shutil.copy2(src, dst)
            else:
                try:
                    dst.symlink_to(src.resolve())
                except OSError:  # Windows without developer mode
                    shutil.copy2(src, dst)
        (out_dir / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(frame_to_yolo_lines(fr)), encoding="utf-8")
        n += 1
    log.info("%s: converted %d frames", split, n)
    return n


def write_dataset_yaml(out_dir: Path) -> Path:
    p = out_dir / "dataset.yaml"
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    p.write_text(f"path: {out_dir.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n",
                 encoding="utf-8")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/bdd100k")
    ap.add_argument("--out", default="data/bdd100k/yolo")
    ap.add_argument("--subset", type=int, default=None, help="max frames per split")
    ap.add_argument("--copy-images", action="store_true")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    for split in ("train", "val"):
        convert_split(root / "labels" / "det_20" / f"det_{split}.json",
                      root / "images" / split, out, split, a.subset, a.copy_images)
    log.info("wrote %s", write_dataset_yaml(out))


if __name__ == "__main__":
    main()
