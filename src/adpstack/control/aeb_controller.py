"""Longitudinal + lateral control: AEB (TTC based), ACC (PID with time gap), lane keeping (Stanley)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adpstack.fusion.risk import Risk, RiskLevel


@dataclass
class Control:
    throttle: float
    brake: float
    steer: float
    aeb_active: bool = False


class PID:
    def __init__(self, kp: float, ki: float, kd: float, out_lim: tuple[float, float] = (-1.0, 1.0)) -> None:
        self.kp, self.ki, self.kd, self.lim = kp, ki, kd, out_lim
        self.i, self.prev = 0.0, None

    def step(self, err: float, dt: float) -> float:
        self.i = float(np.clip(self.i + err * dt, -5, 5))
        d = 0.0 if self.prev is None else (err - self.prev) / max(dt, 1e-3)
        self.prev = err
        return float(np.clip(self.kp * err + self.ki * self.i + self.kd * d, *self.lim))


def stanley_steer(lateral_offset_m: float, heading_err_rad: float, speed_mps: float,
                  k: float = 1.2, softening: float = 1.0, max_steer: float = 0.7) -> float:
    """Stanley law: steer = heading_err + atan(k * e / (v + eps)); output normalised to [-1, 1]."""
    delta = heading_err_rad + np.arctan2(k * lateral_offset_m, speed_mps + softening)
    return float(np.clip(delta, -max_steer, max_steer) / max_steer)


class DrivingController:
    """Combines AEB > ACC > cruise. AEB overrides everything when risk is CRITICAL."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        a = cfg["acc"]
        self.pid = PID(a["kp"], a["ki"], a["kd"])

    def step(self, speed_mps: float, risk: Risk, lead_speed_mps: float | None, lane_offset_m: float,
             heading_err_rad: float, dt: float) -> Control:
        aeb, acc, lka = self.cfg["aeb"], self.cfg["acc"], self.cfg["lka"]
        steer = stanley_steer(lane_offset_m, heading_err_rad, speed_mps, lka["k"], lka["softening"], lka["max_steer"])
        if risk.level == RiskLevel.CRITICAL:
            # brake harder the smaller the TTC
            ttc = max(risk.ttc_s, 0.1) if np.isfinite(risk.ttc_s) else 0.1
            b = float(np.clip(aeb["max_brake"] * (aeb["ttc_brake"] / ttc), 0.5, aeb["max_brake"]))
            return Control(0.0, b, steer, True)
        target = acc["target_speed_mps"]
        if risk.level >= RiskLevel.CAUTION and np.isfinite(risk.distance_m):
            desired_gap = max(aeb["min_dist_m"], acc["time_gap_s"] * speed_mps)
            gap_err = risk.distance_m - desired_gap
            target = float(np.clip(
                (lead_speed_mps if lead_speed_mps is not None else speed_mps) + 0.3 * gap_err, 0.0, target))
        u = self.pid.step(target - speed_mps, dt)
        if risk.level == RiskLevel.WARNING:
            u = min(u, -0.2)                 # gentle pre-brake
        return Control(max(u, 0.0), max(-u, 0.0), steer, False)
