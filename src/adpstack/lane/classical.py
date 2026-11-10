"""Classical lane detection: color/gradient thresholds -> bird's-eye view -> sliding window -> polyfit."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class LaneResult:
    left_fit: np.ndarray | None
    right_fit: np.ndarray | None
    curvature_m: float | None
    offset_m: float | None
    mask: np.ndarray          # HxW uint8 lane-area mask in the ORIGINAL image frame
    confidence: float


def threshold_lanes(bgr: np.ndarray) -> np.ndarray:
    """Binary lane-pixel map from HLS (white + yellow), LAB-b (yellow) and Sobel-x."""
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    white = cv2.inRange(hls, (0, 190, 0), (255, 255, 255))
    yellow = cv2.inRange(hls, (15, 80, 100), (35, 255, 255))
    lab_b = cv2.inRange(lab[:, :, 2], 145, 200)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    sx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    sx = np.uint8(255 * sx / (sx.max() + 1e-6))
    grad = cv2.inRange(sx, 40, 255)
    color = cv2.bitwise_or(cv2.bitwise_or(white, yellow), lab_b)
    out = cv2.bitwise_or(color, cv2.bitwise_and(grad, cv2.dilate(color, np.ones((9, 9), np.uint8))))
    return (out > 0).astype(np.uint8) * 255


def perspective_matrices(shape: tuple[int, int], src_frac: list[list[float]], warp_size: tuple[int, int]
                         ) -> tuple[np.ndarray, np.ndarray]:
    """(M, Minv) mapping the road trapezoid (fractions of image) to a rectangular BEV."""
    h, w = shape
    src = np.float32([[x * w, y * h] for x, y in src_frac])
    ww, wh = warp_size
    dst = np.float32([[ww * 0.2, 0], [ww * 0.8, 0], [ww * 0.8, wh], [ww * 0.2, wh]])
    return cv2.getPerspectiveTransform(src, dst), cv2.getPerspectiveTransform(dst, src)


def sliding_window(binary: np.ndarray, nwindows: int = 9, margin: int = 70, minpix: int = 50
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find left/right lane pixels in a BEV binary image. Returns (lx, ly, rx, ry)."""
    h, w = binary.shape
    hist = np.sum(binary[h // 2:] > 0, axis=0)
    mid = w // 2
    lx_cur = int(np.argmax(hist[:mid])) if hist[:mid].any() else mid // 2
    rx_cur = int(np.argmax(hist[mid:])) + mid if hist[mid:].any() else mid + mid // 2
    nz_y, nz_x = binary.nonzero()
    win_h = h // nwindows
    l_idx, r_idx = [], []
    for i in range(nwindows):
        y_lo, y_hi = h - (i + 1) * win_h, h - i * win_h
        for cur, store in ((lx_cur, l_idx), (rx_cur, r_idx)):
            good = ((nz_y >= y_lo) & (nz_y < y_hi) & (nz_x >= cur - margin) & (nz_x < cur + margin)).nonzero()[0]
            store.append(good)
            if len(good) > minpix:
                if store is l_idx:
                    lx_cur = int(nz_x[good].mean())
                else:
                    rx_cur = int(nz_x[good].mean())
    li = np.concatenate(l_idx) if l_idx else np.array([], int)
    ri = np.concatenate(r_idx) if r_idx else np.array([], int)
    return nz_x[li], nz_y[li], nz_x[ri], nz_y[ri]


def fit_poly(x: np.ndarray, y: np.ndarray, min_pts: int = 150) -> np.ndarray | None:
    """Fit x = a*y^2 + b*y + c; None if too few points."""
    if len(x) < min_pts:
        return None
    return np.polyfit(y, x, 2)


def curvature_and_offset(left: np.ndarray, right: np.ndarray, img_h: int, img_w: int,
                         xm: float, ym: float) -> tuple[float, float]:
    """Radius of curvature (m, at the car) and lateral offset (m, +right of lane centre)."""
    ys = np.linspace(0, img_h - 1, img_h)
    lx, rx = np.polyval(left, ys), np.polyval(right, ys)
    cl = np.polyfit(ys * ym, lx * xm, 2)
    cr = np.polyfit(ys * ym, rx * xm, 2)
    y_eval = (img_h - 1) * ym
    rad = lambda c: (1 + (2 * c[0] * y_eval + c[1]) ** 2) ** 1.5 / (abs(2 * c[0]) + 1e-9)  # noqa: E731
    curvature = float((rad(cl) + rad(cr)) / 2)
    centre = (lx[-1] + rx[-1]) / 2.0
    offset = float((img_w / 2.0 - centre) * xm)
    return curvature, offset


class ClassicalLaneDetector:
    """Stateful detector with exponential temporal smoothing of the polynomial fits."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.alpha = cfg.get("smooth_alpha", 0.8)
        self._left: np.ndarray | None = None
        self._right: np.ndarray | None = None
        self._M: tuple[np.ndarray, np.ndarray] | None = None
        self._shape: tuple[int, int] | None = None

    def _matrices(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        if self._M is None or self._shape != shape:
            self._M = perspective_matrices(shape, self.cfg["src_pts_frac"], tuple(self.cfg["warp_size"]))
            self._shape = shape
        return self._M

    def _smooth(self, old: np.ndarray | None, new: np.ndarray | None) -> np.ndarray | None:
        if new is None:
            return old
        return new if old is None else self.alpha * old + (1 - self.alpha) * new

    def __call__(self, bgr: np.ndarray) -> LaneResult:
        h, w = bgr.shape[:2]
        M, Minv = self._matrices((h, w))
        ww, wh = self.cfg["warp_size"]
        binary = threshold_lanes(bgr)
        bev = cv2.warpPerspective(binary, M, (ww, wh), flags=cv2.INTER_LINEAR)
        lx, ly, rx, ry = sliding_window(bev, self.cfg["nwindows"], self.cfg["margin"], self.cfg["minpix"])
        lf, rf = fit_poly(lx, ly), fit_poly(rx, ry)
        self._left, self._right = self._smooth(self._left, lf), self._smooth(self._right, rf)
        mask = np.zeros((h, w), np.uint8)
        curv = off = None
        conf = (lf is not None) * 0.5 + (rf is not None) * 0.5
        if self._left is not None and self._right is not None:
            ys = np.linspace(0, wh - 1, wh)
            pl = np.stack([np.polyval(self._left, ys), ys], 1)
            pr = np.stack([np.polyval(self._right, ys), ys], 1)[::-1]
            poly = np.vstack([pl, pr]).astype(np.float32)
            bev_mask = np.zeros((wh, ww), np.uint8)
            cv2.fillPoly(bev_mask, [poly.astype(np.int32)], 255)
            mask = cv2.warpPerspective(bev_mask, Minv, (w, h))
            curv, off = curvature_and_offset(self._left, self._right, wh, ww,
                                             self.cfg["xm_per_pix"], self.cfg["ym_per_pix"])
        return LaneResult(self._left, self._right, curv, off, mask, conf)
