import cv2
import numpy as np

from adpstack.detection.error_analysis import export_error_grids
from adpstack.detection.infer import Detections


class _SyntheticDetector:
    def __call__(self, _image):
        return Detections(
            xyxy=np.array([[10, 10, 30, 30], [40, 40, 50, 50]], dtype=np.float32),
            conf=np.array([0.9, 0.99], dtype=np.float32),
            cls=np.array([0, 1], dtype=np.int32),
        )


def test_export_error_grids_marks_a_missed_object_and_false_positive(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    image_path = images_dir / "frame.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((100, 100, 3), dtype=np.uint8))
    frame = {
        "name": image_path.name,
        "labels": [
            {"category": "car", "box2d": {"x1": 10, "y1": 10, "x2": 30, "y2": 30}},
            {"category": "person", "box2d": {"x1": 60, "y1": 60, "x2": 80, "y2": 80}},
        ],
    }

    fn_path, fp_path = export_error_grids(
        [frame], images_dir, _SyntheticDetector(), out_dir=tmp_path / "figures"
    )

    assert fn_path.name == "errors_fn.jpg"
    assert fp_path.name == "errors_fp.jpg"
    for grid_path in (fn_path, fp_path):
        grid = cv2.imread(str(grid_path))
        assert grid is not None
        assert grid.shape == (180, 320, 3)
        assert np.any(grid)
