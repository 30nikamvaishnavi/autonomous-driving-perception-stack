import numpy as np

from adpstack.tracking.bytetrack_wrapper import ByteTracker, KalmanBox, iou_matrix
from adpstack.tracking.metrics import mot_metrics


def test_iou():
    a = np.array([[0, 0, 10, 10]], float)
    assert iou_matrix(a, a)[0, 0] > 0.999
    assert iou_matrix(a, np.array([[20, 20, 30, 30]], float))[0, 0] == 0
    assert iou_matrix(np.zeros((0, 4)), a).shape == (0, 1)


def test_kalman_tracks_constant_velocity():
    kf = KalmanBox(np.array([0, 0, 20, 40], float))
    for i in range(1, 15):
        kf.predict()
        kf.update(np.array([5 * i, 0, 20 + 5 * i, 40], float))
    pred = kf.predict()
    assert abs(pred[0] - 75) < 3


def _seq(n=30, with_noise=False):
    rng = np.random.default_rng(0)
    for t in range(n):
        boxes = np.array([[10 + 4 * t, 100, 60 + 4 * t, 160], [400 - 3 * t, 200, 450 - 3 * t, 260]], float)
        if with_noise:
            boxes += rng.normal(0, 1, boxes.shape)
        yield boxes


def test_ids_persist():
    tr = ByteTracker(min_hits=2)
    ids = []
    for b in _seq(with_noise=True):
        out = tr.update(b, np.array([0.9, 0.9]), np.array([0, 0]))
        ids.append(sorted(t.id for t in out))
    assert ids[-1] == [1, 2] and all(i == [1, 2] for i in ids[3:])


def test_low_conf_recovered_and_occlusion_buffer():
    tr = ByteTracker(min_hits=1, track_buffer=10)
    seq = list(_seq(12))
    for b in seq[:6]:
        tr.update(b, np.array([0.9, 0.9]), np.array([0, 0]))
    out = tr.update(seq[6], np.array([0.3, 0.9]), np.array([0, 0]))   # low-conf second stage
    assert sorted(t.id for t in out) == [1, 2]
    for _ in seq[7:9]:                                                 # fully missing detections
        tr.update(np.zeros((0, 4)), np.zeros(0), np.zeros(0))
    out = tr.update(seq[9], np.array([0.9, 0.9]), np.array([0, 0]))
    assert sorted(t.id for t in out) == [1, 2]


def test_metrics_perfect_and_switch():
    gt = [(np.array([1, 2]), b) for b in _seq(10)]
    perfect = [(np.array([1, 2]), b) for b in _seq(10)]
    m = mot_metrics(gt, perfect)
    assert m["MOTA"] == 1.0 and m["IDF1"] == 1.0 and m["ID_switches"] == 0
    sw = [(np.array([1, 2]) if i < 5 else np.array([1, 3]), b) for i, b in enumerate(_seq(10))]
    m2 = mot_metrics(gt, sw)
    assert m2["ID_switches"] == 1 and m2["IDF1"] < 1.0 and m2["MOTA"] < 1.0
