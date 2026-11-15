"""Fine-tune an Ultralytics YOLO model on BDD100K (YOLO format) with W&B logging."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from adpstack.utils.config import load_config, resolve_device
from adpstack.utils.logging import get_logger
from adpstack.utils.seed import seed_everything

log = get_logger(__name__)


def build_train_args(cfg: dict) -> dict:
    """Map our YAML config to Ultralytics ``model.train`` kwargs."""
    dev = resolve_device(cfg.get("device", "auto"))
    aug = cfg.get("augment", {})
    return dict(
        data=cfg["data"], epochs=cfg["epochs"], imgsz=cfg["imgsz"], batch=cfg["batch"],
        device=0 if dev == "cuda" else dev, workers=cfg.get("workers", 4),
        project=cfg["project"], name=cfg["name"], patience=cfg.get("patience", 15),
        cos_lr=cfg.get("cos_lr", True), seed=cfg.get("seed", 42), plots=True,
        save_period=cfg.get("save_period", -1), **aug,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/yolo.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. epochs=5 model=yolov8s.pt")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    seed_everything(int(cfg.get("seed", 42)))
    from ultralytics import YOLO, settings

    wb = cfg.get("wandb", {})
    if wb.get("enabled", True):
        os.environ.setdefault("WANDB_PROJECT", wb.get("project", "adpstack"))
        settings.update({"wandb": True})
    else:
        settings.update({"wandb": False})
    model = YOLO(cfg["model"])
    args = build_train_args(cfg)
    log.info("training with %s", args)

    if wb.get("enabled", True) and os.environ.get("WANDB_MODE") != "offline":
        url_path = Path(cfg["project"]) / cfg["name"] / "wandb_run_url.txt"

        def save_wandb_url(_trainer) -> None:
            import wandb

            if wandb.run is not None:
                url_path.parent.mkdir(parents=True, exist_ok=True)
                url_path.write_text(wandb.run.url, encoding="utf-8")
                log.info("W&B run URL: %s", wandb.run.url)

        model.add_callback("on_train_start", save_wandb_url)

    model.train(resume=a.resume, **args) if a.resume else model.train(**args)
    log.info("best weights: %s/%s/weights/best.pt", cfg["project"], cfg["name"])


if __name__ == "__main__":
    main()
