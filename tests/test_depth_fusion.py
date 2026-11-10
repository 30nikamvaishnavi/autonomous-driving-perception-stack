import numpy as np

from adpstack.control.aeb_controller import DrivingController, stanley_steer
from adpstack.depth.calibrate_scale import (depth_metrics, fit_scale_shift, midas_to_metric, pinhole_distance)
from adpstack.fusion.distance_estimator import DistanceEstimator, sample_box_depth
from adpstack.fusion.risk import RiskLevel, Risk, classify, scene_risk
from adpstack.fusion.ttc import DistanceKalman, TTCEstimator, time_to_collision
from adpstack.utils.config import load_config


def test_scale_shift_recovers_known_params():
    rng = np.random.default_rng(1)
    gt = rng.uniform(3, 60, (60, 80))
    s, t = 12.0, 0.02
    midas = (1.0 / gt - t) / s
    s2, t2 = fit_scale_shift(midas, gt)
    assert abs(s2 - s) / s < 1e-3 and abs(t2 - t) < 1e-4
    rec = midas_to_metric(midas, s2, t2)
    m = depth_metrics(rec, gt)
    assert m["AbsRel"] < 1e-3 and m["delta1"] == 1.0


def test_pinhole():
    assert abs(pinhole_distance(100, 1.5, 960) - 14.4) < 1e-9


def test_sample_box_depth_median():
    d = np.full((100, 100), 10.0)
    d[:20] = 99
    assert sample_box_depth(d, np.array([20, 40, 80, 90.0])) == 10.0
    assert np.isnan(sample_box_depth(d, np.array([50, 50, 50, 50.0])))


def test_distance_estimator_fuses():
    cfg = load_config("configs/depth.yaml")
    est = DistanceEstimator(cfg)
    gt = np.full((720, 1280), 15.0)
    box = np.array([600, 300, 700, 400.0])        # 100 px tall car -> pinhole 14.4 m
    r = est.estimate(box, "car", metric_depth=gt)
    assert r.method == "fused" and 14.4 < r.distance_m < 15.0
    assert est.estimate(box, "traffic sign").method == "none"
    assert est.estimate(box, "car").method == "pinhole"


def test_kalman_ttc_closing_speed():
    est = TTCEstimator()
    out = None
    for i in range(30):
        t = i * 0.05
        out = est.update(1, 30.0 - 10.0 * t, t)        # closing at 10 m/s
    d, v, ttc = out
    assert abs(v + 10.0) < 1.0 and abs(ttc - d / 10.0) < 0.5
    assert time_to_collision(10, +2.0) == float("inf")
    assert time_to_collision(10, -5.0) == 2.0
    assert DistanceKalman(5.0).update(5.0, 0.0) == (5.0, 0.0)


def test_risk_levels_and_corridor():
    assert classify(30, 10) == RiskLevel.SAFE
    assert classify(30, 2.5) == RiskLevel.WARNING
    assert classify(30, 1.0) == RiskLevel.CRITICAL
    assert classify(3, float("inf")) == RiskLevel.CRITICAL
    objs = [{"id": 1, "distance_m": 5, "lateral_m": 6.0, "ttc_s": 0.5},     # outside corridor
            {"id": 2, "distance_m": 20, "lateral_m": 0.2, "ttc_s": 2.5}]
    r = scene_risk(objs)
    assert r.track_id == 2 and r.level == RiskLevel.WARNING


def test_controller_aeb_overrides_and_cruise():
    cfg = load_config("configs/control.yaml")
    c = DrivingController(cfg)
    crit = Risk(RiskLevel.CRITICAL, 1, 5.0, 1.0)
    out = c.step(8.0, crit, 0.0, 0.0, 0.0, 0.05)
    assert out.aeb_active and out.brake >= 0.5 and out.throttle == 0
    safe = Risk(RiskLevel.SAFE, None, float("inf"), float("inf"))
    out = DrivingController(cfg).step(2.0, safe, None, 0.0, 0.0, 0.05)
    assert out.throttle > 0 and out.brake == 0 and not out.aeb_active
    assert stanley_steer(1.0, 0.0, 5.0) > 0 > stanley_steer(-1.0, 0.0, 5.0)
