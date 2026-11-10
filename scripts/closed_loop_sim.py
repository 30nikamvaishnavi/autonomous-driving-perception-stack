"""Kinematic closed-loop check of the AEB/ACC logic (no CARLA, no GPU).

Ego drives at 14 m/s toward a stationary obstacle (or a braking lead car). Distance is estimated by
the pinhole model from a *rendered box height* with noise, filtered by the TTC Kalman filter, classified
by risk, and fed to the controller. Reports stopping distance / collision. This validates the control
logic only; it says nothing about perception accuracy (see docs/results.md for measured numbers).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from adpstack.control.aeb_controller import DrivingController
from adpstack.fusion.risk import Risk, RiskLevel, scene_risk
from adpstack.fusion.ttc import TTCEstimator
from adpstack.utils.config import load_config


def simulate(aeb_on: bool = True, v0: float = 14.0, gap0: float = 60.0, lead_v: float = 0.0,
             noise_px: float = 1.5, seed: int = 0, dt: float = 0.05, max_decel: float = 8.0,
             latency_s: float = 0.1) -> dict:
    rng = np.random.default_rng(seed)
    cfg = load_config("configs/control.yaml")
    fy, h_real = 960.0, 1.5
    ctrl, ttc = DrivingController(cfg), TTCEstimator()
    ego_v, gap, t = v0, gap0, 0.0
    hist, pending = [], []
    min_gap = gap0
    while t < 20.0:
        true_h = fy * h_real / max(gap, 0.5)
        meas_d = fy * h_real / max(true_h + rng.normal(0, noise_px), 1.0)
        d, v, tt = ttc.update(1, meas_d, t)
        risk = scene_risk([{"id": 1, "distance_m": d, "lateral_m": 0.0, "ttc_s": tt}])
        if not aeb_on:
            risk = Risk(RiskLevel.SAFE, None, math.inf, math.inf)
        pending.append((t + latency_s, ctrl.step(ego_v, risk, lead_v, 0.0, 0.0, dt)))
        cmd = [c for ts, c in pending if ts <= t][-1] if any(ts <= t for ts, _ in pending) else None
        thr, brk = (cmd.throttle, cmd.brake) if cmd else (0.0, 0.0)
        acc = 3.0 * thr - max_decel * brk
        ego_v = max(0.0, ego_v + acc * dt)
        gap += (lead_v - ego_v) * dt
        min_gap = min(min_gap, gap)
        hist.append((t, ego_v, gap, brk))
        if gap <= 0:
            return {"collision": True, "impact_speed_mps": ego_v, "min_gap_m": 0.0, "history": hist}
        if ego_v < 0.05 and lead_v == 0.0:
            break
        t += dt
    return {"collision": False, "impact_speed_mps": 0.0, "min_gap_m": float(min_gap), "history": hist}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/figures")
    a = ap.parse_args()
    rows = {}
    for name, kw in {"stationary_obstacle": dict(lead_v=0.0, gap0=60.0),
                     "slow_lead_5mps": dict(lead_v=5.0, gap0=45.0)}.items():
        for aeb in (False, True):
            runs = [simulate(aeb, seed=s, **kw) for s in range(20)]
            rows[f"{name}|aeb={aeb}"] = {
                "collision_rate": float(np.mean([r["collision"] for r in runs])),
                "mean_min_gap_m": float(np.mean([r["min_gap_m"] for r in runs])),
                "mean_impact_speed_mps": float(np.mean([r["impact_speed_mps"] for r in runs]))}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "closed_loop_sim.json").write_text(json.dumps(rows, indent=2))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for aeb, col in ((False, "tab:red"), (True, "tab:green")):
        h = np.array(simulate(aeb, seed=0)["history"])
        ax[0].plot(h[:, 0], h[:, 1], color=col, label=f"AEB {'on' if aeb else 'off'}")
        ax[1].plot(h[:, 0], h[:, 2], color=col)
    ax[0].set(xlabel="t [s]", ylabel="ego speed [m/s]", title="Speed")
    ax[1].set(xlabel="t [s]", ylabel="gap to obstacle [m]", title="Gap")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(out / "closed_loop_sim.png", dpi=130)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
