import json

import numpy as np

from adpstack.data.bdd100k_to_yolo import (
    box_to_yolo,
    convert_split,
    frame_to_yolo_lines,
    write_dataset_yaml,
)
from adpstack.data.carla_recorder import (
    boxes_from_semantic_instance,
    carla_depth_to_meters,
    intrinsics,
)
from adpstack.data.eda import compute_stats, imbalance_report, plot_all, size_buckets

FRAME = {"name": "a.jpg", "attributes": {"weather": "rainy", "timeofday": "night", "scene": "city street"},
         "labels": [
             {"category": "car", "attributes": {"occluded": True, "truncated": False},
              "box2d": {"x1": 100, "y1": 200, "x2": 300, "y2": 400}},
             {"category": "person", "box2d": {"x1": 0, "y1": 0, "x2": 64, "y2": 72}},
             {"category": "lane", "poly2d": []},
             {"category": "car", "box2d": {"x1": 5, "y1": 5, "x2": 5.2, "y2": 5.2}}]}


def test_box_to_yolo_center_and_clip():
    cx, cy, w, h = box_to_yolo(0, 0, 1280, 720)
    assert (cx, cy, w, h) == (0.5, 0.5, 1.0, 1.0)
    assert box_to_yolo(-50, -50, 10, 10)[2] == 10 / 1280
    assert box_to_yolo(10, 10, 10.5, 10.5) is None


def test_frame_to_lines_filters_classes_and_degenerate():
    lines = frame_to_yolo_lines(FRAME)
    assert len(lines) == 2
    assert lines[0].startswith("0 ") and lines[1].startswith("5 ")


def test_frame_to_lines_normalizes_category_aliases_and_skips_unknown():
    frame = {
        "labels": [
            {"category": "bike", "box2d": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}},
            {"category": "motor", "box2d": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}},
            {"category": "lane", "box2d": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}},
        ]
    }

    lines = frame_to_yolo_lines(frame)

    assert [line.split()[0] for line in lines] == ["4", "3"]
    stats = compute_stats([frame])
    assert stats["class_counts"]["bicycle"] == 1
    assert stats["class_counts"]["motorcycle"] == 1


def test_convert_split(tmp_path):
    root = tmp_path / "bdd"
    (root / "images").mkdir(parents=True)
    (root / "images" / "a.jpg").write_bytes(b"x")
    lj = tmp_path / "l.json"
    lj.write_text(json.dumps([FRAME, {"name": "missing.jpg", "labels": []}]))
    n = convert_split(lj, root / "images", tmp_path / "out", "train", copy_images=True)
    assert n == 1
    assert (tmp_path / "out/labels/train/a.txt").read_text().count("\n") == 1
    y = write_dataset_yaml(tmp_path / "out")
    assert "traffic light" in y.read_text()


def test_eda_stats_and_imbalance(tmp_path):
    st = compute_stats([FRAME])
    assert st["class_counts"]["car"] == 2 and st["timeofday"]["night"] == 1
    rep = imbalance_report({"car": 1000, "bus": 10})
    assert rep["imbalance_ratio"] == 100
    assert rep["class_weights"]["bus"] > rep["class_weights"]["car"]
    assert size_buckets([100, 50 * 50, 200 * 200]) == {"small(<32^2)": 1, "medium": 1, "large(>96^2)": 1}
    assert len(plot_all(st, tmp_path)) == 4


def test_carla_helpers():
    K = intrinsics(1280, 720, 90)
    assert abs(K[0, 0] - 640) < 1e-6 and K[0, 2] == 640
    bgra = np.zeros((2, 2, 4), np.uint8)
    bgra[..., 2] = 255   # R=255 -> 255/(256^3-1)*1000 m
    assert abs(carla_depth_to_meters(bgra)[0, 0] - 255 / (256**3 - 1) * 1000) < 1e-6
    sem = np.zeros((100, 100), int)
    inst = np.zeros((100, 100), int)
    sem[10:30, 20:60] = 14
    inst[10:30, 20:60] = 7
    boxes = boxes_from_semantic_instance(sem, inst)
    assert len(boxes) == 1 and boxes[0][0] == 0
    assert abs(boxes[0][3] - 0.4) < 1e-9
