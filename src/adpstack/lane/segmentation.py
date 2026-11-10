"""Deep-learning lane / drivable-area segmentation (U-Net + MobileNetV2) trained on BDD100K drivable maps.

BDD100K drivable-area masks: 0 = directly drivable (ego lane), 1 = alternatively drivable, 2 = background.
Train on Colab (GPU):  python -m adpstack.lane.segmentation train --images ... --masks ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from adpstack.utils.config import load_config, resolve_device
from adpstack.utils.logging import get_logger

log = get_logger(__name__)
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def preprocess(bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """BGR HxWx3 -> normalised NCHW float32 at ``size``=(h, w)."""
    img = cv2.resize(bgr, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)[:, :, ::-1] / 255.0
    return ((img - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)


def postprocess(logits: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """(1,C,h,w) logits -> HxW class-id map (uint8) resized to the original image."""
    cls = logits[0].argmax(0).astype(np.uint8)
    return cv2.resize(cls, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_NEAREST)


def build_model(encoder: str = "mobilenet_v2", classes: int = 3, pretrained: bool = True):
    import segmentation_models_pytorch as smp

    return smp.Unet(encoder, encoder_weights="imagenet" if pretrained else None, classes=classes)


class SegLaneDetector:
    """Inference wrapper (PyTorch .pt state-dict or ONNX). Returns an ego-lane mask + class map."""

    def __init__(self, weights: str, cfg: dict, device: str = "auto") -> None:
        self.size = tuple(cfg["seg"]["imgsz"])
        self.onnx = str(weights).endswith(".onnx")
        if self.onnx:
            import onnxruntime as ort

            self.sess = ort.InferenceSession(weights, providers=["CPUExecutionProvider"])
            self.inp = self.sess.get_inputs()[0].name
        else:
            import torch

            self.device = resolve_device(device)
            self.model = build_model(cfg["seg"]["encoder"], cfg["seg"]["classes"], pretrained=False)
            self.model.load_state_dict(torch.load(weights, map_location=self.device))
            self.model.to(self.device).eval()

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        x = preprocess(bgr, self.size)
        if self.onnx:
            logits = self.sess.run(None, {self.inp: x})[0]
        else:
            import torch

            with torch.no_grad():
                logits = self.model(torch.from_numpy(x).to(self.device)).cpu().numpy()
        return postprocess(logits, bgr.shape[:2])

    @staticmethod
    def ego_lane_mask(class_map: np.ndarray) -> np.ndarray:
        return ((class_map == 0) * 255).astype(np.uint8)


def train(images: Path, masks: Path, cfg: dict, epochs: int, batch: int, out: Path,
          wandb_on: bool = True) -> None:  # pragma: no cover - needs GPU + data
    import torch
    from torch.utils.data import DataLoader, Dataset

    from adpstack.utils.wandb_logger import WandbLogger

    size = tuple(cfg["seg"]["imgsz"])

    class DS(Dataset):
        def __init__(self, files: list[Path], train_mode: bool) -> None:
            self.files, self.train_mode = files, train_mode

        def __len__(self) -> int:
            return len(self.files)

        def __getitem__(self, i: int):
            img = cv2.imread(str(self.files[i]))
            m = cv2.imread(str(masks / (self.files[i].stem + ".png")), cv2.IMREAD_GRAYSCALE)
            m = cv2.resize(m, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
            x = preprocess(img, size)[0]
            if self.train_mode and np.random.rand() < 0.5:
                x, m = x[:, :, ::-1].copy(), m[:, ::-1].copy()
            return torch.from_numpy(x), torch.from_numpy(m.astype(np.int64))

    files = sorted(p for p in images.glob("*.jpg") if (masks / (p.stem + ".png")).exists())
    k = int(0.9 * len(files))
    tr = DataLoader(DS(files[:k], True), batch_size=batch, shuffle=True, num_workers=2)
    va = DataLoader(DS(files[k:], False), batch_size=batch, num_workers=2)
    dev = resolve_device("auto")
    model = build_model(cfg["seg"]["encoder"], cfg["seg"]["classes"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    wb = WandbLogger(name="seg-unet-mnv2", config=cfg, enabled=wandb_on, tags=["lane", "seg"])
    best = 0.0
    for ep in range(epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            loss = torch.nn.functional.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        ious = []
        with torch.no_grad():
            for x, y in va:
                pred = model(x.to(dev)).argmax(1).cpu().numpy()
                for p, t in zip(pred, y.numpy(), strict=True):
                    ious.append(mask_iou(p == 0, t == 0))
        miou = float(np.mean(ious))
        wb.log({"epoch": ep, "train_loss": float(loss), "val_ego_iou": miou})
        if miou > best:
            best = miou
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out)
    wb.finish()


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Binary IoU (1.0 when both masks are empty)."""
    a, b = a.astype(bool), b.astype(bool)
    u = (a | b).sum()
    return 1.0 if u == 0 else float((a & b).sum() / u)


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--config", default="configs/lane.yaml")
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="runs/seg/best.pt")
    a = ap.parse_args()
    cfg = load_config(a.config)
    train(Path(a.images), Path(a.masks), cfg, a.epochs, a.batch, Path(a.out))


if __name__ == "__main__":
    main()
