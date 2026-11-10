import numpy as np

from adpstack.detection.infer import Detections, decode_yolo_output, letterbox
from adpstack.fusion.distance_estimator import DistanceEstimator
from adpstack.pipeline import PerceptionPipeline
from adpstack.tracking.bytetrack_wrapper import ByteTracker
from adpstack.utils.config import load_config
from adpstack.utils.timing import Timer, latency_stats
from adpstack.viz.bev import draw_bev
from adpstack.viz.overlay import draw_hud, draw_tracks


def test_letterbox_shape_and_mapping():
    img = np.zeros((720, 1280, 3), np.uint8)
    out, r, (px, py) = letterbox(img, 640)
    assert out.shape == (640, 640, 3) and r == 0.5 and px == 0 and py == 140


def test_decode_yolo_output_inverts_letterbox_and_nms():
    nc = 9
    pred = np.zeros((1, 4 + nc, 3), np.float32)
    # box centred (320, 320) 100x100 in letterboxed space; duplicate at near-identical location; one low conf
    pred[0, :4, 0] = [320, 320, 100, 100]
    pred[0, 4, 0] = 0.9
    pred[0, :4, 1] = [322, 321, 100, 100]
    pred[0, 4, 1] = 0.8
    pred[0, :4, 2] = [50, 50, 20, 20]
    pred[0, 6, 2] = 0.1
    det = decode_yolo_output(pred, 0.5, (0, 140), (720, 1280), conf_thr=0.25)
    assert len(det) == 1 and det.cls[0] == 0
    x1, y1, x2, y2 = det.xyxy[0]
    assert abs(x1 - 540) < 1 and abs(x2 - 740) < 1 and abs(y1 - 260) < 1 and abs(y2 - 460) < 1
    assert len(decode_yolo_output(np.zeros((1, 13, 5), np.float32), 1, (0, 0), (10, 10))) == 0


def test_end_to_end_pipeline_brakes_for_approaching_car():
    class FakeDet:
        def __init__(self):
            self.t = 0

        def __call__(self, img):
            self.t += 1
            h = 40 + 4 * self.t                         # car grows -> approaches
            return Detections(np.array([[600, 400 - h, 600 + 1.6 * h, 400]], np.float32),
                              np.array([0.9], np.float32), np.array([0], np.int32))

    cfg = load_config("configs/depth.yaml")
    pipe = PerceptionPipeline(FakeDet(), ByteTracker(min_hits=1), DistanceEstimator(cfg))
    levels, last = [], None
    for i in range(60):
        last = pipe.process(np.zeros((720, 1280, 3), np.uint8), i * 0.05)
        levels.append(int(last.risk.level))
    assert last.objects and last.objects[0]["distance_m"] < 15
    assert last.objects[0]["rel_speed_mps"] < 0
    assert max(levels) >= 2 and "detect" in pipe.timer.summary()
    vis = draw_hud(draw_tracks(np.zeros((720, 1280, 3), np.uint8), last.objects), levels[-1], brake=0.9)
    assert vis.shape == (720, 1280, 3) and draw_bev(last.objects).shape == (400, 300, 3)


def test_timing_stats():
    s = latency_stats([10.0] * 10)
    assert s["mean_ms"] == 10 and s["fps"] == 100 and s["p99_ms"] == 10
    t = Timer()
    with t.measure("x"):
        pass
    assert len(t.records["x"]) == 1
