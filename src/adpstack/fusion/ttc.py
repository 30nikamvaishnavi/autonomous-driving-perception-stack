"""Kalman-smoothed distance / closing-speed per track and time-to-collision."""
from __future__ import annotations

import numpy as np


class DistanceKalman:
    """1-D constant-velocity Kalman filter on [distance, d_distance/dt]."""

    def __init__(self, d0: float, q: float = 1.0, r: float = 1.5) -> None:
        self.x = np.array([d0, 0.0])
        self.P = np.diag([r, 25.0])
        self.q, self.r = q, r
        self.t: float | None = None

    def update(self, d: float, t: float) -> tuple[float, float]:
        if self.t is None:
            self.t = t
            return float(self.x[0]), float(self.x[1])
        dt = max(t - self.t, 1e-3)
        self.t = t
        F = np.array([[1, dt], [0, 1]])
        Q = self.q * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        H = np.array([[1.0, 0.0]])
        S = H @ self.P @ H.T + self.r
        K = self.P @ H.T / S
        self.x = self.x + (K * (d - H @ self.x)).ravel()
        self.P = (np.eye(2) - K @ H) @ self.P
        return float(self.x[0]), float(self.x[1])


def time_to_collision(distance_m: float, rel_speed_mps: float, min_closing: float = 0.3) -> float:
    """TTC in seconds. ``rel_speed`` = d(distance)/dt, so closing means negative. inf if not closing."""
    closing = -rel_speed_mps
    if closing < min_closing or distance_m <= 0:
        return float("inf")
    return float(distance_m / closing)


class TTCEstimator:
    """Tracks per-ID filters and returns (smoothed_distance, rel_speed, ttc)."""

    def __init__(self, max_idle_s: float = 2.0) -> None:
        self.filters: dict[int, DistanceKalman] = {}
        self.last_seen: dict[int, float] = {}
        self.max_idle_s = max_idle_s

    def update(self, track_id: int, distance_m: float, t: float) -> tuple[float, float, float]:
        kf = self.filters.setdefault(track_id, DistanceKalman(distance_m))
        d, v = kf.update(distance_m, t)
        self.last_seen[track_id] = t
        for tid in [k for k, ts in self.last_seen.items() if t - ts > self.max_idle_s]:
            self.filters.pop(tid, None)
            self.last_seen.pop(tid, None)
        return d, v, time_to_collision(d, v)
