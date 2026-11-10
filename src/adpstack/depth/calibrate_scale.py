"""Convert MiDaS relative inverse depth to metric depth.

(a) Fit  1/depth_m = s * midas + t  against ground-truth depth (CARLA) by least squares.
(b) Pinhole known-height prior: Z = fy * H_real / h_pixels.
"""
from __future__ import annotations

import numpy as np


def fit_scale_shift(midas: np.ndarray, gt_depth_m: np.ndarray, min_d: float = 1.0, max_d: float = 80.0
                    ) -> tuple[float, float]:
    """Least-squares (scale, shift) mapping MiDaS output to inverse metric depth (1/m), with outlier trimming."""
    m = midas.ravel().astype(np.float64)
    g = gt_depth_m.ravel().astype(np.float64)
    ok = (g > min_d) & (g < max_d) & np.isfinite(m)
    m, inv = m[ok], 1.0 / g[ok]
    if m.size < 10:
        raise ValueError("not enough valid pixels to fit")
    A = np.stack([m, np.ones_like(m)], 1)
    s, t = np.linalg.lstsq(A, inv, rcond=None)[0]
    for _ in range(2):                                   # trim worst 10 %, refit
        res = np.abs(A @ np.array([s, t]) - inv)
        keep = res < np.percentile(res, 90)
        s, t = np.linalg.lstsq(A[keep], inv[keep], rcond=None)[0]
    return float(s), float(t)


def fit_over_dataset(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
    """Fit a single (scale, shift) over many (midas, gt) pairs."""
    m = np.concatenate([p[0].ravel() for p in pairs])
    g = np.concatenate([p[1].ravel() for p in pairs])
    return fit_scale_shift(m, g)


def midas_to_metric(midas: np.ndarray, scale: float, shift: float, min_inv: float = 1e-3) -> np.ndarray:
    """Apply (scale, shift) and invert to meters."""
    inv = np.maximum(scale * midas + shift, min_inv)
    return 1.0 / inv


def pinhole_distance(box_height_px: float, real_height_m: float, fy: float) -> float:
    """Distance from the apparent height of an object of known real height."""
    return fy * real_height_m / max(box_height_px, 1e-6)


def depth_metrics(pred: np.ndarray, gt: np.ndarray, min_d: float = 1.0, max_d: float = 80.0) -> dict[str, float]:
    """MAE, RMSE, AbsRel, delta<1.25 over valid GT pixels."""
    ok = (gt > min_d) & (gt < max_d) & np.isfinite(pred)
    p, g = pred[ok].astype(np.float64), gt[ok].astype(np.float64)
    if g.size == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "AbsRel": float("nan"), "delta1": float("nan")}
    ratio = np.maximum(p / g, g / p)
    return {"MAE": float(np.abs(p - g).mean()), "RMSE": float(np.sqrt(((p - g) ** 2).mean())),
            "AbsRel": float((np.abs(p - g) / g).mean()), "delta1": float((ratio < 1.25).mean())}
