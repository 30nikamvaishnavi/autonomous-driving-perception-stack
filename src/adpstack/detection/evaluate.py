"""Evaluate a YOLO model: mAP50, mAP50-95, per-class AP, PR curves, confusion matrix, failure analysis."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from adpstack.data.bdd100k_to_yolo import CLASSES
from adpstack.utils.logging import get_logger

log = get_logger(__name__)


def split_by_attribute(frames: list[dict], images_dir: Path) -> dict[str, list[str]]:
    """Group image paths by BDD100K attribute slices used for failure analysis."""
    groups: dict[str, list[str]] = {"night": [], "day": [], "rainy": [], "clear": [], "small_objects": []}
    for fr in frames:
        p = str(images_dir / fr["name"])
        a = fr.get("attributes", {})
        groups["night" if a.get("timeofday") == "night" else "day"].append(p)
        groups["rainy" if a.get("weather") in ("rainy", "snowy") else "clear"].append(p)
        areas = [(b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) for lab in fr.get("labels") or []
                 if (b := lab.get("box2d"))]
        if areas and sum(x < 32**2 for x in areas) / len(areas) > 0.6:
            groups["small_objects"].append(p)
    return groups


def _val(model, data: str, imgsz: int, device: str, **kw) -> dict:
    m = model.val(data=data, imgsz=imgsz, device=device, plots=kw.pop("plots", False), verbose=False, **kw)
    return {"mAP50": float(m.box.map50), "mAP50-95": float(m.box.map),
            "precision": float(m.box.mp), "recall": float(m.box.mr),
            "per_class_AP50-95": {CLASSES[int(i)]: float(v) for i, v in
                                  zip(m.box.ap_class_index, m.box.maps[m.box.ap_class_index], strict=True)}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data/bdd100k/yolo/dataset.yaml")
    ap.add_argument("--labels-json", default="data/bdd100k/labels/det_20/det_val.json")
    ap.add_argument("--images-dir", default="data/bdd100k/yolo/images/val")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="docs/figures/detection_eval.json")
    a = ap.parse_args()
    from ultralytics import YOLO

    model = YOLO(a.weights)
    res = {"overall": _val(model, a.data, a.imgsz, a.device, plots=True)}  # saves PR curves + confusion matrix
    with open(a.labels_json, encoding="utf-8") as f:
        frames = json.load(f)
    base = Path(a.data).parent
    for name, imgs in split_by_attribute(frames, Path(a.images_dir)).items():
        if len(imgs) < 20:
            continue
        with tempfile.TemporaryDirectory() as td:
            lst = Path(td) / "list.txt"
            lst.write_text("\n".join(imgs))
            y = Path(td) / "d.yaml"
            y.write_text(f"path: {base.resolve().as_posix()}\ntrain: {lst.as_posix()}\nval: {lst.as_posix()}\n"
                         + "names:\n" + "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES)) + "\n")
            res[name] = _val(model, str(y), a.imgsz, a.device)
            res[name]["n_images"] = len(imgs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    log.info("saved %s", a.out)


if __name__ == "__main__":
    main()
