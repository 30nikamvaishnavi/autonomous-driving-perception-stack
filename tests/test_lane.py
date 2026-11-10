import cv2
import numpy as np

from adpstack.lane.classical import ClassicalLaneDetector, fit_poly, perspective_matrices, threshold_lanes
from adpstack.lane.compare import f1_score
from adpstack.lane.segmentation import mask_iou, postprocess, preprocess
from adpstack.utils.config import load_config


def synthetic_road(w=1280, h=720, curve=0):
    img = np.full((h, w, 3), 70, np.uint8)
    cv2.rectangle(img, (0, 0), (w, int(h * 0.55)), (200, 180, 150), -1)       # sky
    horizon = int(h * 0.55)
    vp = (w // 2 + curve, horizon)
    for bx, col in ((int(w * 0.18), (255, 255, 255)), (int(w * 0.82), (0, 220, 255))):
        cv2.line(img, (bx, h), vp, col, 14)
    return img


def test_threshold_finds_white_and_yellow():
    b = threshold_lanes(synthetic_road())
    assert b[600:, :].sum() > 0 and b[:300].sum() == 0


def test_perspective_roundtrip():
    cfg = load_config("configs/lane.yaml")
    M, Minv = perspective_matrices((720, 1280), cfg["src_pts_frac"], tuple(cfg["warp_size"]))
    p = np.array([[[640.0, 500.0]]], np.float32)
    back = cv2.perspectiveTransform(cv2.perspectiveTransform(p, M), Minv)
    assert np.allclose(back, p, atol=1e-2)


def test_fit_poly_requires_points():
    assert fit_poly(np.arange(10), np.arange(10)) is None
    y = np.arange(500)
    f = fit_poly(2e-4 * y**2 + 0.1 * y + 50, y, 100)
    assert abs(f[0] - 2e-4) < 1e-6


def test_classical_detector_finds_lane_area():
    cfg = load_config("configs/lane.yaml")
    det = ClassicalLaneDetector(cfg)
    img = synthetic_road()
    for _ in range(3):
        res = det(img)
    assert res.left_fit is not None and res.right_fit is not None
    assert res.mask.shape == img.shape[:2] and res.mask.sum() > 0
    assert res.offset_m is not None and abs(res.offset_m) < 2.0
    # the detected area should sit between the two painted lines near the bottom of the image
    row = res.mask[int(720 * 0.9)] > 0
    xs = np.nonzero(row)[0]
    assert xs.size and 0.15 * 1280 < xs.min() < 0.5 * 1280 and 0.5 * 1280 < xs.max() < 0.9 * 1280


def test_seg_pre_post_and_metrics():
    x = preprocess(np.zeros((720, 1280, 3), np.uint8), (384, 640))
    assert x.shape == (1, 3, 384, 640) and x.dtype == np.float32
    logits = np.zeros((1, 3, 96, 160), np.float32)
    logits[0, 1] = 1
    assert postprocess(logits, (720, 1280)).shape == (720, 1280)
    a = np.zeros((10, 10), bool)
    a[:5] = True
    assert mask_iou(a, a) == 1.0 and mask_iou(a, ~a) == 0.0 and mask_iou(~a & a, ~a & a) == 1.0
    assert f1_score(a, a) == 1.0
