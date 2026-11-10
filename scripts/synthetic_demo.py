"""Model-free demo: synthetic road + approaching car -> full fusion/risk/control stack -> mp4.

Needs only numpy/opencv. Useful to verify the install and to generate a quick preview video without
weights, GPU or CARLA. The 'detector' is a fake that reports the rendered car box; lane detection,
tracking, distance/TTC, risk and HUD run for real.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from adpstack.detection.infer import Detections
from adpstack.fusion.distance_estimator import DistanceEstimator
from adpstack.lane.classical import ClassicalLaneDetector
from adpstack.pipeline import PerceptionPipeline
from adpstack.tracking.bytetrack_wrapper import ByteTracker
from adpstack.utils.config import load_config
from adpstack.viz.bev import draw_bev
from adpstack.viz.overlay import draw_hud, draw_lane, draw_tracks


def render(gap: float, w=1280, h=720):
    img = np.full((h, w, 3), 85, np.uint8)
    cv2.rectangle(img, (0, 0), (w, int(h * 0.55)), (210, 190, 160), -1)
    vp = (w // 2, int(h * 0.55))
    for bx, col in ((int(w * 0.18), (255, 255, 255)), (int(w * 0.82), (0, 220, 255))):
        cv2.line(img, (bx, h), vp, col, 14)
    bh = int(960 * 1.5 / max(gap, 1.0))
    bw = int(bh * 1.6)
    cy = int(h * 0.55) + bh // 2 + int(25 * (h * 0.45) / max(gap, 1.0) / 3)
    box = np.array([w / 2 - bw / 2, cy - bh, w / 2 + bw / 2, cy], np.float32)
    cv2.rectangle(img, tuple(box[:2].astype(int)), tuple(box[2:].astype(int)), (30, 30, 200), -1)
    return img, box


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/synthetic_demo.mp4")
    a = ap.parse_args()
    state = {"gap": 45.0, "box": None}

    def fake_det(_):
        return Detections(state["box"][None], np.array([0.9], np.float32), np.array([0], np.int32))

    pipe = PerceptionPipeline(fake_det, ByteTracker(min_hits=1), DistanceEstimator(load_config("configs/depth.yaml")),
                              ClassicalLaneDetector(load_config("configs/lane.yaml")))
    from adpstack.control.aeb_controller import DrivingController

    ctrl = DrivingController(load_config("configs/control.yaml"))
    speed, dt = 12.0, 0.05
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    wr = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), 20, (1280 + 300, 720))
    for i in range(240):
        img, state["box"] = render(state["gap"])
        res = pipe.process(img, i * dt)
        c = ctrl.step(speed, res.risk, 0.0, res.lane.offset_m or 0.0, 0.0, dt)
        speed = max(0.0, speed + (3 * c.throttle - 8 * c.brake) * dt)
        state["gap"] = max(2.0, state["gap"] - speed * dt)
        vis = draw_hud(draw_tracks(draw_lane(img, res.lane.mask), res.objects), int(res.risk.level),
                       speed_mps=speed, lane_offset_m=res.lane.offset_m, curvature_m=res.lane.curvature_m,
                       brake=c.brake)
        bev = cv2.resize(draw_bev(res.objects), (300, 720))
        wr.write(np.hstack([vis, bev]))
        if speed < 0.05:
            break
    wr.release()
    print(f"wrote {a.out}; final gap {state['gap']:.1f} m, speed {speed:.2f} m/s")


if __name__ == "__main__":
    main()
