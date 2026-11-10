"""Bird's-eye-view radar plot of tracked objects (ego at bottom centre)."""
from __future__ import annotations

import cv2
import numpy as np


def draw_bev(objs: list[dict], size: tuple[int, int] = (300, 400), max_range_m: float = 40.0,
             half_width_m: float = 8.0) -> np.ndarray:
    """objs: dict(id, distance_m, lateral_m). Returns a BGR image."""
    w, h = size
    img = np.full((h, w, 3), 30, np.uint8)
    sx, sy = w / (2 * half_width_m), (h - 20) / max_range_m
    for r in range(10, int(max_range_m) + 1, 10):
        y = int(h - 10 - r * sy)
        cv2.line(img, (0, y), (w, y), (70, 70, 70), 1)
        cv2.putText(img, f"{r}m", (4, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    for x in (-1.8, 1.8):
        px = int(w / 2 + x * sx)
        cv2.line(img, (px, 0), (px, h), (0, 120, 0), 1)
    cv2.rectangle(img, (w // 2 - 8, h - 22), (w // 2 + 8, h - 6), (255, 255, 255), -1)
    for o in objs:
        d, lat = o["distance_m"], o["lateral_m"]
        if not np.isfinite(d) or d > max_range_m or abs(lat) > half_width_m:
            continue
        p = (int(w / 2 + lat * sx), int(h - 10 - d * sy))
        cv2.circle(img, p, 6, (0, 140, 255), -1)
        cv2.putText(img, str(o["id"]), (p[0] + 7, p[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return img
