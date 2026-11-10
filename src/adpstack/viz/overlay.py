"""Drawing utilities: detections, tracks with distance/TTC, lane mask, HUD."""
from __future__ import annotations

import cv2
import numpy as np

PALETTE = [(255, 128, 0), (0, 200, 255), (0, 255, 128), (255, 0, 128), (180, 0, 255), (0, 128, 255),
           (128, 255, 0), (255, 255, 0), (0, 255, 255)]
RISK_COLORS = {0: (0, 200, 0), 1: (0, 220, 220), 2: (0, 140, 255), 3: (0, 0, 255)}


def draw_detections(img: np.ndarray, det, names: list[str]) -> np.ndarray:
    out = img.copy()
    for b, c, k in zip(det.xyxy, det.conf, det.cls, strict=True):
        col = PALETTE[int(k) % len(PALETTE)]
        x1, y1, x2, y2 = (int(v) for v in b)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.putText(out, f"{names[int(k)]} {c:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    return out


def draw_tracks(img: np.ndarray, objs: list[dict]) -> np.ndarray:
    """objs: dict(id, xyxy, name, distance_m, ttc_s)."""
    out = img.copy()
    for o in objs:
        col = PALETTE[o["id"] % len(PALETTE)]
        x1, y1, x2, y2 = (int(v) for v in o["xyxy"])
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        d, ttc = o.get("distance_m"), o.get("ttc_s")
        label = f"#{o['id']} {o['name']}"
        if d is not None and np.isfinite(d):
            label += f" {d:.1f}m"
        if ttc is not None and np.isfinite(ttc):
            label += f" TTC {ttc:.1f}s"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), col, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def draw_lane(img: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha: float = 0.35) -> np.ndarray:
    out = img.copy()
    m = mask > 0
    out[m] = ((1 - alpha) * out[m] + alpha * np.array(color)).astype(np.uint8)
    return out


def draw_hud(img: np.ndarray, risk_level: int, speed_mps: float | None = None, lane_offset_m: float | None = None,
             curvature_m: float | None = None, fps: float | None = None, brake: float | None = None) -> np.ndarray:
    out = img.copy()
    col = RISK_COLORS[int(risk_level)]
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.circle(out, (20, 17), 9, col, -1)
    txt = [f"RISK {['SAFE', 'CAUTION', 'WARNING', 'CRITICAL'][int(risk_level)]}"]
    if speed_mps is not None:
        txt.append(f"{speed_mps * 3.6:.0f} km/h")
    if lane_offset_m is not None:
        txt.append(f"offset {lane_offset_m:+.2f} m")
    if curvature_m is not None:
        txt.append(f"R {min(curvature_m, 9999):.0f} m")
    if brake is not None:
        txt.append(f"brake {brake:.2f}")
    if fps is not None:
        txt.append(f"{fps:.1f} FPS")
    cv2.putText(out, "  |  ".join(txt), (38, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if brake and brake > 0.5:
        cv2.putText(out, "AEB BRAKING", (out.shape[1] // 2 - 110, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
    return out
