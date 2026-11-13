"""Export ranked false-negative and false-positive image grids."""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from adpstack.data.bdd100k_to_yolo import CLASS_TO_ID, CLASSES
from adpstack.detection.infer import Detections, Detector

THUMBNAIL_SIZE = (320, 180)


@dataclass(frozen=True)
class _Error:
    image_path: Path
    box: tuple[float, float, float, float]
    class_name: str
    rank: float


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    second_area = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _class_name(class_id: int) -> str:
    return CLASSES[class_id] if 0 <= class_id < len(CLASSES) else f"class_{class_id}"


def _collect_frame_errors(
    image_path: Path,
    image: np.ndarray,
    frame: dict,
    detector: Callable[[np.ndarray], Detections],
    iou_threshold: float,
) -> tuple[list[_Error], list[_Error]]:
    ground_truth = []
    for label in frame.get("labels") or []:
        category = label.get("category")
        box = label.get("box2d")
        class_id = CLASS_TO_ID.get(category)
        if class_id is None or not box:
            continue
        coordinates = np.array([box["x1"], box["y1"], box["x2"], box["y2"]], dtype=float)
        if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
            continue
        ground_truth.append((coordinates, class_id))

    detections = detector(image)
    matched: set[int] = set()
    false_positives = []
    for detection_index in np.argsort(detections.conf)[::-1]:
        detection_box = detections.xyxy[detection_index]
        detection_class = int(detections.cls[detection_index])
        best_index, best_iou = None, 0.0
        for gt_index, (gt_box, gt_class) in enumerate(ground_truth):
            if gt_index in matched or gt_class != detection_class:
                continue
            overlap = _iou(detection_box, gt_box)
            if overlap > best_iou:
                best_index, best_iou = gt_index, overlap
        if best_index is not None and best_iou >= iou_threshold:
            matched.add(best_index)
        else:
            score = float(detections.conf[detection_index])
            false_positives.append(
                _Error(
                    image_path=image_path,
                    box=tuple(float(value) for value in detection_box),
                    class_name=_class_name(detection_class),
                    rank=score,
                )
            )

    false_negatives = []
    for gt_index, (gt_box, class_id) in enumerate(ground_truth):
        if gt_index not in matched:
            area = max(0.0, float(gt_box[2] - gt_box[0])) * max(
                0.0, float(gt_box[3] - gt_box[1])
            )
            false_negatives.append(
                _Error(
                    image_path=image_path,
                    box=tuple(float(value) for value in gt_box),
                    class_name=_class_name(class_id),
                    rank=area,
                )
            )
    return false_negatives, false_positives


def _write_grid(errors: list[_Error], path: Path, title: str, limit: int, columns: int) -> None:
    selected = errors[:limit]
    thumb_width, thumb_height = THUMBNAIL_SIZE
    grid_columns = min(columns, max(1, len(selected)))
    rows = max(1, math.ceil(len(selected) / grid_columns))
    canvas = np.zeros((rows * thumb_height, grid_columns * thumb_width, 3), dtype=np.uint8)

    for index, error in enumerate(selected):
        image = cv2.imread(str(error.image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image for error grid: {error.image_path}")
        original_height, original_width = image.shape[:2]
        thumbnail = cv2.resize(image, THUMBNAIL_SIZE, interpolation=cv2.INTER_AREA)
        x_scale, y_scale = thumb_width / original_width, thumb_height / original_height
        x1, y1, x2, y2 = error.box
        box = (
            round(x1 * x_scale),
            round(y1 * y_scale),
            round(x2 * x_scale),
            round(y2 * y_scale),
        )
        color = (0, 0, 255) if title == "FN" else (0, 165, 255)
        cv2.rectangle(thumbnail, box[:2], box[2:], color, 2)
        detail = f"{title} {error.class_name} rank={error.rank:.2f}"
        cv2.putText(thumbnail, detail, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        row, column = divmod(index, grid_columns)
        y, x = row * thumb_height, column * thumb_width
        canvas[y:y + thumb_height, x:x + thumb_width] = thumbnail

    if not selected:
        cv2.putText(canvas, f"No {title} errors found", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Could not write error-analysis grid: {path}")


def export_error_grids(
    frames: Sequence[dict],
    images_dir: Path,
    detector: Callable[[np.ndarray], Detections],
    out_dir: Path = Path("docs/figures"),
    limit: int = 50,
    iou_threshold: float = 0.5,
    columns: int = 5,
) -> tuple[Path, Path]:
    """Write up to ``limit`` ranked false negatives and false positives as JPEG grids.

    False negatives are ranked smallest-box-first; false positives are ranked by
    descending detector confidence. Matches require the same class and IoU threshold.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    if columns < 1:
        raise ValueError("columns must be positive")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {images_dir}")

    false_negatives, false_positives = [], []
    for frame in frames:
        image_path = images_dir / frame["name"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        missed, extra = _collect_frame_errors(image_path, image, frame, detector, iou_threshold)
        false_negatives.extend(missed)
        false_positives.extend(extra)

    false_negatives.sort(key=lambda error: (error.rank, str(error.image_path), error.class_name))
    false_positives.sort(key=lambda error: (-error.rank, str(error.image_path), error.class_name))
    fn_path, fp_path = out_dir / "errors_fn.jpg", out_dir / "errors_fp.jpg"
    _write_grid(false_negatives, fn_path, "FN", limit, columns)
    _write_grid(false_positives, fp_path, "FP", limit, columns)
    return fn_path, fp_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--labels-json", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--out-dir", default="docs/figures")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with Path(args.labels_json).open(encoding="utf-8") as label_file:
        frames = json.load(label_file)
    detector = Detector(args.weights, device=args.device)
    fn_path, fp_path = export_error_grids(
        frames, Path(args.images_dir), detector, Path(args.out_dir), args.limit, args.iou
    )
    print(f"False-negative grid: {fn_path}")
    print(f"False-positive grid: {fp_path}")


if __name__ == "__main__":
    main()
