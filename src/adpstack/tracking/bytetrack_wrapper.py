"""ByteTrack multi-object tracker (numpy + scipy; Kalman filter, two-stage association)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between xyxy arrays (N,4) and (M,4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


class KalmanBox:
    """Constant-velocity Kalman filter over state [cx, cy, a, h, vx, vy, va, vh]."""

    def __init__(self, xyxy: np.ndarray) -> None:
        self.x = np.zeros(8)
        self.x[:4] = self._to_z(xyxy)
        self.P = np.diag([10, 10, 1e-2, 10, 1e3, 1e3, 1e-1, 1e3]).astype(float)
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)
        self.Q = np.diag([1, 1, 1e-4, 1, 1e-1, 1e-1, 1e-6, 1e-1]).astype(float)
        self.R = np.diag([4, 4, 1e-2, 4]).astype(float)

    @staticmethod
    def _to_z(b: np.ndarray) -> np.ndarray:
        w, h = b[2] - b[0], b[3] - b[1]
        return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h])

    @property
    def xyxy(self) -> np.ndarray:
        cx, cy, a, h = self.x[:4]
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.xyxy

    def update(self, xyxy: np.ndarray) -> None:
        z = self._to_z(xyxy)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P


@dataclass
class Track:
    id: int
    kf: KalmanBox
    cls: int
    conf: float
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    confirmed: bool = False

    @property
    def xyxy(self) -> np.ndarray:
        return self.kf.xyxy


def _associate(tracks: list[Track], boxes: np.ndarray, thr: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if not tracks or len(boxes) == 0:
        return [], list(range(len(tracks))), list(range(len(boxes)))
    cost = 1.0 - iou_matrix(np.array([t.xyxy for t in tracks]), boxes)
    r, c = linear_sum_assignment(cost)
    matches = [(i, j) for i, j in zip(r, c, strict=True) if cost[i, j] <= thr]
    mi, mj = {i for i, _ in matches}, {j for _, j in matches}
    return matches, [i for i in range(len(tracks)) if i not in mi], [j for j in range(len(boxes)) if j not in mj]


class ByteTracker:
    """ByteTrack (Zhang et al., 2022): associates high-confidence boxes first, then low-confidence ones."""

    def __init__(self, track_high_thresh: float = 0.5, track_low_thresh: float = 0.1,
                 new_track_thresh: float = 0.6, track_buffer: int = 30, match_thresh: float = 0.8,
                 min_hits: int = 2) -> None:
        self.hi, self.lo, self.new_thr = track_high_thresh, track_low_thresh, new_track_thresh
        self.buffer, self.match_thr, self.min_hits = track_buffer, match_thresh, min_hits
        self.tracks: list[Track] = []
        self._next_id = 1
        self.frame = 0

    def reset(self) -> None:
        self.tracks, self._next_id, self.frame = [], 1, 0

    def update(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> list[Track]:
        """Feed one frame of detections; returns confirmed tracks seen this frame."""
        self.frame += 1
        xyxy = np.asarray(xyxy, float).reshape(-1, 4)
        conf, cls = np.asarray(conf, float), np.asarray(cls, int)
        for t in self.tracks:
            t.kf.predict()
            t.age += 1
            t.time_since_update += 1
        hi_m, lo_m = conf >= self.hi, (conf >= self.lo) & (conf < self.hi)
        hi_idx, lo_idx = np.nonzero(hi_m)[0], np.nonzero(lo_m)[0]

        m1, un_t, un_h = _associate(self.tracks, xyxy[hi_idx], self.match_thr)
        for ti, dj in m1:
            self._apply(self.tracks[ti], hi_idx[dj], xyxy, conf, cls)
        rest = [self.tracks[i] for i in un_t]
        m2, un_t2, _ = _associate(rest, xyxy[lo_idx], 0.5)
        for ti, dj in m2:
            self._apply(rest[ti], lo_idx[dj], xyxy, conf, cls)
        for dj in un_h:
            k = hi_idx[dj]
            if conf[k] >= self.new_thr:
                t = Track(self._next_id, KalmanBox(xyxy[k]), int(cls[k]), float(conf[k]))
                t.confirmed = self.min_hits <= 1
                self._next_id += 1
                self.tracks.append(t)
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.buffer]
        return [t for t in self.tracks if t.confirmed and t.time_since_update == 0]

    def _apply(self, t: Track, k: int, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        t.kf.update(xyxy[k])
        t.conf, t.cls = float(conf[k]), int(cls[k])
        t.hits += 1
        t.time_since_update = 0
        if t.hits >= self.min_hits:
            t.confirmed = True
