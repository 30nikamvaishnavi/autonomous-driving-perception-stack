"""End-to-end perception pipeline: detect -> track -> lane -> depth -> distance/TTC -> risk."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from adpstack.data.bdd100k_to_yolo import CLASSES
from adpstack.fusion.distance_estimator import DistanceEstimator
from adpstack.fusion.risk import Risk, scene_risk
from adpstack.fusion.ttc import TTCEstimator
from adpstack.lane.classical import LaneResult
from adpstack.tracking.bytetrack_wrapper import ByteTracker
from adpstack.utils.timing import Timer


@dataclass
class FrameResult:
    objects: list[dict] = field(default_factory=list)
    lane: LaneResult | None = None
    risk: Risk | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    depth: np.ndarray | None = None


class PerceptionPipeline:
    """Components are injected so the pipeline works with PyTorch, ONNX or TensorRT backends (and fakes in tests)."""

    def __init__(self, detector: Callable[[np.ndarray], Any], tracker: ByteTracker,
                 distance: DistanceEstimator, lane: Callable[[np.ndarray], LaneResult] | None = None,
                 depth: Callable[[np.ndarray], np.ndarray] | None = None, class_names: list[str] | None = None,
                 depth_every: int = 1, risk_cfg: dict | None = None) -> None:
        self.detector, self.tracker, self.distance = detector, tracker, distance
        self.lane, self.depth = lane, depth
        self.names = class_names or CLASSES
        self.ttc = TTCEstimator()
        self.timer = Timer()
        self.depth_every = depth_every
        self.risk_cfg = risk_cfg or {}
        self._n = 0
        self._last_depth: np.ndarray | None = None

    def process(self, bgr: np.ndarray, t: float, metric_depth: np.ndarray | None = None) -> FrameResult:
        self._n += 1
        res = FrameResult()
        with self.timer.measure("detect"):
            det = self.detector(bgr)
        with self.timer.measure("track"):
            tracks = self.tracker.update(det.xyxy, det.conf, det.cls)
        if self.lane is not None:
            with self.timer.measure("lane"):
                res.lane = self.lane(bgr)
        rel = None
        if metric_depth is None and self.depth is not None:
            if self._n % self.depth_every == 1 or self.depth_every == 1 or self._last_depth is None:
                with self.timer.measure("depth"):
                    self._last_depth = self.depth(bgr)
            rel = self._last_depth
        res.depth = rel
        with self.timer.measure("fusion"):
            for tr in tracks:
                name = self.names[tr.cls]
                est = self.distance.estimate(tr.xyxy, name, midas_rel=rel, metric_depth=metric_depth)
                if not np.isfinite(est.distance_m):
                    continue
                d, v, ttc = self.ttc.update(tr.id, est.distance_m, t)
                res.objects.append({"id": tr.id, "name": name, "xyxy": tr.xyxy.copy(), "conf": tr.conf,
                                    "distance_m": d, "raw_distance_m": est.distance_m, "lateral_m": est.lateral_m,
                                    "rel_speed_mps": v, "ttc_s": ttc, "method": est.method})
            res.risk = scene_risk(res.objects, **self.risk_cfg)
        res.timings_ms = {k: v[-1] for k, v in self.timer.records.items()}
        return res
