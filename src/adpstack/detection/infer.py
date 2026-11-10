"""Backend-agnostic detector: Ultralytics .pt, ONNX Runtime (.onnx; CPU/DirectML/CUDA/TensorRT EP)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from adpstack.data.bdd100k_to_yolo import CLASSES


@dataclass
class Detections:
    """Detections for one frame: xyxy in original pixel coordinates."""
    xyxy: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    conf: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.float32))
    cls: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int32))

    def __len__(self) -> int:
        return len(self.conf)


def letterbox(img: np.ndarray, size: int = 640, color: int = 114
              ) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize with aspect ratio preserved and pad to square. Returns (img, scale, (padx, pady))."""
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padx, pady = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), color, np.uint8)
    out[pady:pady + nh, padx:padx + nw] = resized
    return out, r, (padx, pady)


def decode_yolo_output(pred: np.ndarray, scale: float, pad: tuple[int, int],
                       orig_shape: tuple[int, int], conf_thr: float = 0.25,
                       iou_thr: float = 0.5) -> Detections:
    """Decode a YOLOv8/11 raw ONNX output (1, 4+nc, N) into NMS-filtered Detections."""
    p = (pred[0] if pred.ndim == 3 else pred).T      # Ultralytics layout (4+nc, N) -> (N, 4+nc)
    boxes, scores = p[:, :4], p[:, 4:]
    cls = scores.argmax(1)
    conf = scores[np.arange(len(scores)), cls]
    keep = conf >= conf_thr
    boxes, conf, cls = boxes[keep], conf[keep], cls[keep]
    if len(conf) == 0:
        return Detections()
    cx, cy, w, h = boxes.T
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad[0]) / scale
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad[1]) / scale
    H, W = orig_shape
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, W)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, H)
    xywh = np.stack([xyxy[:, 0], xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]], 1)
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), conf.tolist(), conf_thr, iou_thr)
    idx = np.array(idx).reshape(-1)
    return Detections(xyxy[idx].astype(np.float32), conf[idx].astype(np.float32), cls[idx].astype(np.int32))


class Detector:
    """Unified interface: ``Detector('best.pt')`` or ``Detector('best.onnx', providers=[...])``."""

    def __init__(self, weights: str, imgsz: int = 640, conf: float = 0.25, iou: float = 0.5,
                 device: str = "cpu", providers: list[str] | None = None,
                 class_names: list[str] | None = None) -> None:
        self.imgsz, self.conf, self.iou = imgsz, conf, iou
        self.names = class_names or CLASSES
        self.backend = "onnx" if str(weights).endswith(".onnx") else "torch"
        if self.backend == "onnx":
            import onnxruntime as ort

            self.sess = ort.InferenceSession(str(weights), providers=providers or ["CPUExecutionProvider"])
            self.inp = self.sess.get_inputs()[0].name
            shape = self.sess.get_inputs()[0].shape
            if isinstance(shape[2], int):
                self.imgsz = shape[2]
        else:
            from ultralytics import YOLO

            self.model = YOLO(str(weights))
            self.device = device
            if hasattr(self.model, "names"):
                self.names = [self.model.names[i] for i in sorted(self.model.names)]

    def __call__(self, bgr: np.ndarray) -> Detections:
        if self.backend == "torch":
            r = self.model.predict(bgr, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                   device=self.device, verbose=False)[0]
            b = r.boxes
            return Detections(b.xyxy.cpu().numpy().astype(np.float32), b.conf.cpu().numpy().astype(np.float32),
                              b.cls.cpu().numpy().astype(np.int32))
        img, scale, pad = letterbox(bgr, self.imgsz)
        x = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.sess.run(None, {self.inp: np.ascontiguousarray(x)})[0]
        return decode_yolo_output(out, scale, pad, bgr.shape[:2], self.conf, self.iou)


def available_providers() -> list[str]:
    import onnxruntime as ort

    return ort.get_available_providers()


def main() -> None:
    import argparse

    from adpstack.viz.overlay import draw_detections

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="outputs/det.jpg")
    a = ap.parse_args()
    img = cv2.imread(a.image)
    det = Detector(a.weights)(img)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, draw_detections(img, det, CLASSES))


if __name__ == "__main__":
    main()
