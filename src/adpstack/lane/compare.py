"""Compare classical vs segmentation lane detectors: IoU, F1, FPS, per-condition tables, qualitative grid."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from adpstack.lane.classical import ClassicalLaneDetector
from adpstack.lane.segmentation import SegLaneDetector, mask_iou
from adpstack.utils.config import load_config
from adpstack.utils.timing import latency_stats


def f1_score(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    tp = (p & g).sum()
    denom = 2 * tp + (p & ~g).sum() + (~p & g).sum()
    return 1.0 if denom == 0 else float(2 * tp / denom)


def evaluate(detect_fn, images: list[Path], masks_dir: Path) -> dict:
    ious, f1s, lat = [], [], []
    for p in images:
        img = cv2.imread(str(p))
        gt = cv2.imread(str(masks_dir / f"{p.stem}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            continue
        t0 = time.perf_counter()
        pred = detect_fn(img) > 0
        lat.append((time.perf_counter() - t0) * 1000)
        ego = gt == 0
        ious.append(mask_iou(pred, ego))
        f1s.append(f1_score(pred, ego))
    return {"IoU": float(np.mean(ious)) if ious else 0.0, "F1": float(np.mean(f1s)) if f1s else 0.0,
            "n": len(ious), **latency_stats(lat)}


def qualitative_grid(images: list[Path], fns: dict, out: Path, size=(320, 180)) -> None:
    rows = []
    for p in images:
        img = cv2.imread(str(p))
        cells = [cv2.resize(img, size)]
        for name, fn in fns.items():
            m = fn(img) > 0
            ov = img.copy()
            ov[m] = (0.5 * ov[m] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
            c = cv2.resize(ov, size)
            cv2.putText(c, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cells.append(c)
        rows.append(np.hstack(cells))
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows))


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--seg-weights", default=None)
    ap.add_argument("--config", default="configs/lane.yaml")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--out", default="docs/figures")
    a = ap.parse_args()
    cfg = load_config(a.config)
    imgs = sorted(Path(a.images).glob("*.jpg"))[: a.limit]
    classical = ClassicalLaneDetector(cfg)
    fns = {"classical": lambda im: classical(im).mask}
    if a.seg_weights:
        seg = SegLaneDetector(a.seg_weights, cfg)
        fns["segmentation"] = lambda im: SegLaneDetector.ego_lane_mask(seg(im))
    res = {name: evaluate(fn, imgs, Path(a.masks)) for name, fn in fns.items()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "lane_compare.json").write_text(json.dumps(res, indent=2))
    qualitative_grid(imgs[:6], fns, out / "lane_qualitative.png")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
