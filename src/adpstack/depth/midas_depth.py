"""Monocular depth with MiDaS (torch hub) -> relative inverse depth; ONNX Runtime backend optional."""
from __future__ import annotations

import cv2
import numpy as np

from adpstack.utils.config import resolve_device


class MiDaSDepth:
    """MiDaS_small wrapper. ``__call__`` returns a float32 HxW *relative inverse depth* map."""

    def __init__(self, model_type: str = "MiDaS_small", input_size: int = 256, device: str = "auto",
                 onnx_path: str | None = None) -> None:
        self.size = input_size
        self.onnx = onnx_path is not None
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)
        if self.onnx:
            import onnxruntime as ort

            self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            self.inp = self.sess.get_inputs()[0].name
        else:
            import torch

            self.device = resolve_device(device)
            self.model = torch.hub.load("intel-isl/MiDaS", model_type).to(self.device).eval()

    def _prep(self, bgr: np.ndarray) -> np.ndarray:
        h = self.size
        w = int(round(bgr.shape[1] / bgr.shape[0] * h / 32) * 32)  # keep aspect, multiple of 32
        img = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_CUBIC)[:, :, ::-1].astype(np.float32) / 255.0
        return ((img - self.mean) / self.std).transpose(2, 0, 1)[None].astype(np.float32)

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        x = self._prep(bgr)
        if self.onnx:
            out = self.sess.run(None, {self.inp: x})[0]
        else:
            import torch

            with torch.no_grad():
                out = self.model(torch.from_numpy(x).to(self.device)).cpu().numpy()
        d = np.squeeze(out).astype(np.float32)
        return cv2.resize(d, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
