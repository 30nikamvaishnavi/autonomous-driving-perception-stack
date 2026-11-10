"""Per-object distance from tracked boxes: MiDaS (metric-calibrated) fused with the pinhole height prior."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adpstack.depth.calibrate_scale import midas_to_metric, pinhole_distance


@dataclass
class DistanceEstimate:
    distance_m: float
    lateral_m: float
    method: str
    d_midas: float | None = None
    d_pinhole: float | None = None


def sample_box_depth(depth: np.ndarray, xyxy: np.ndarray, region: str = "lower_half") -> float:
    """Median of the depth map in a robust sub-region of the box (lower half avoids sky/roof)."""
    h, w = depth.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    x1, x2, y1, y2 = max(0, x1), min(w, x2), max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return float("nan")
    if region == "lower_half":
        y1 = (y1 + y2) // 2
    # central 60 % horizontally to avoid background at the box edges
    cw = int((x2 - x1) * 0.2)
    patch = depth[y1:y2, x1 + cw:max(x1 + cw + 1, x2 - cw)]
    return float(np.median(patch)) if patch.size else float("nan")


class DistanceEstimator:
    """Combine MiDaS and pinhole estimates. ``depth_map`` may be None (pinhole only)."""

    def __init__(self, cfg: dict, w_midas: float = 0.5) -> None:
        cam = cfg["camera"]
        self.fx, self.fy, self.cx = cam["fx"], cam["fy"], cam["cx"]
        self.heights = cfg["class_heights_m"]
        self.scale, self.shift = cfg["scale_shift"]["scale"], cfg["scale_shift"]["shift"]
        self.region = cfg.get("sample_region", "lower_half")
        self.w_midas = w_midas

    def estimate(self, xyxy: np.ndarray, class_name: str, midas_rel: np.ndarray | None = None,
                 metric_depth: np.ndarray | None = None) -> DistanceEstimate:
        d_m = d_p = None
        if metric_depth is not None:
            d_m = sample_box_depth(metric_depth, xyxy, self.region)
        elif midas_rel is not None:
            d_m = sample_box_depth(midas_to_metric(midas_rel, self.scale, self.shift), xyxy, self.region)
        if class_name in self.heights:
            d_p = pinhole_distance(float(xyxy[3] - xyxy[1]), self.heights[class_name], self.fy)
        vals = [(d, w) for d, w in ((d_m, self.w_midas), (d_p, 1 - self.w_midas))
                if d is not None and np.isfinite(d) and d > 0]
        if not vals:
            return DistanceEstimate(float("nan"), float("nan"), "none", d_m, d_p)
        z = sum(d * w for d, w in vals) / sum(w for _, w in vals)
        method = "fused" if len(vals) == 2 else ("midas" if d_m in [v[0] for v in vals] else "pinhole")
        u = (xyxy[0] + xyxy[2]) / 2
        return DistanceEstimate(float(z), float((u - self.cx) * z / self.fx), method, d_m, d_p)
