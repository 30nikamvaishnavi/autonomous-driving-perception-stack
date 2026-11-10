"""Exploratory data analysis for BDD100K detection labels."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from adpstack.utils.logging import get_logger  # noqa: E402

log = get_logger(__name__)


def compute_stats(frames: list[dict]) -> dict:
    """Aggregate class counts, boxes/image, box areas, attribute splits, occlusion/truncation."""
    cls = Counter()
    per_img, areas, tod, weather, scene = [], [], Counter(), Counter(), Counter()
    occl, trunc, per_cls_area = Counter(), Counter(), {}
    for fr in frames:
        labs = [lab for lab in fr.get("labels") or [] if lab.get("box2d")]
        per_img.append(len(labs))
        a = fr.get("attributes", {})
        tod[a.get("timeofday", "unknown")] += 1
        weather[a.get("weather", "unknown")] += 1
        scene[a.get("scene", "unknown")] += 1
        for lab in labs:
            c = lab["category"]
            b = lab["box2d"]
            area = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
            cls[c] += 1
            areas.append(area)
            per_cls_area.setdefault(c, []).append(area)
            at = lab.get("attributes", {})
            occl[bool(at.get("occluded"))] += 1
            trunc[bool(at.get("truncated"))] += 1
    total = sum(cls.values())
    return {"n_frames": len(frames), "class_counts": dict(cls), "boxes_per_image": per_img,
            "areas": areas, "per_class_area": per_cls_area, "timeofday": dict(tod),
            "weather": dict(weather), "scene": dict(scene), "occluded": dict(occl),
            "truncated": dict(trunc), "total_boxes": total}


def imbalance_report(class_counts: dict[str, int]) -> dict:
    """Imbalance ratio and inverse-frequency class weights (normalised to mean 1)."""
    if not class_counts:
        return {"imbalance_ratio": 0.0, "class_weights": {}}
    counts = np.array(list(class_counts.values()), dtype=float)
    w = counts.sum() / (len(counts) * counts)
    w = w / w.mean()
    return {"imbalance_ratio": float(counts.max() / counts.min()),
            "class_weights": {k: float(v) for k, v in zip(class_counts, w, strict=True)}}


def size_buckets(areas: list[float]) -> dict[str, int]:
    """COCO-style small/medium/large buckets (pixel area)."""
    a = np.asarray(areas)
    return {"small(<32^2)": int((a < 32**2).sum()),
            "medium": int(((a >= 32**2) & (a < 96**2)).sum()),
            "large(>96^2)": int((a >= 96**2).sum())}


def plot_all(stats: dict, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    def save(fig, name):
        p = out / name
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)

    cc = dict(sorted(stats["class_counts"].items(), key=lambda kv: -kv[1]))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(cc.keys(), cc.values(), color="#2a6fdb")
    ax.set_yscale("log")
    ax.set_title("Class balance (log scale)")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    save(fig, "class_balance.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(stats["boxes_per_image"], bins=40, color="#2a9d8f")
    ax.set_title("Boxes per image")
    save(fig, "boxes_per_image.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.sqrt(np.maximum(stats["areas"], 1)), bins=60, color="#e76f51")
    ax.set_xlabel("sqrt(box area) [px]")
    ax.set_title("Box size distribution")
    save(fig, "box_size.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, key in zip(axes, ("timeofday", "weather", "scene"), strict=True):
        d = stats[key]
        ax.bar(d.keys(), d.values())
        ax.set_title(key)
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    save(fig, "attribute_splits.png")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()
    with open(a.labels, encoding="utf-8") as f:
        frames = json.load(f)
    st = compute_stats(frames)
    rep = imbalance_report(st["class_counts"])
    sz = size_buckets(st["areas"])
    log.info("frames=%d boxes=%d imbalance=%.1f", st["n_frames"], st["total_boxes"],
             rep["imbalance_ratio"])
    paths = plot_all(st, Path(a.out))
    summary = {k: v for k, v in st.items() if k not in ("boxes_per_image", "areas", "per_class_area")}
    summary.update(rep)
    summary["size_buckets"] = sz
    (Path(a.out) / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if a.wandb:
        from adpstack.utils.wandb_logger import WandbLogger

        wb = WandbLogger(name="eda", tags=["eda"])
        for p in paths:
            wb.log_image(p.stem, str(p))
        wb.log({"imbalance_ratio": rep["imbalance_ratio"]})
        wb.finish()


if __name__ == "__main__":
    main()
