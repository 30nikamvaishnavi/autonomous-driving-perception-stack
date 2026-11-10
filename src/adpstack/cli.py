"""Command line entry point: ``adp run --source video.mp4 --weights best.pt --out outputs/demo.mp4``."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from adpstack.utils.config import load_config
from adpstack.utils.logging import get_logger

log = get_logger(__name__)


def build_pipeline(args: argparse.Namespace):
    from adpstack.detection.infer import Detector
    from adpstack.fusion.distance_estimator import DistanceEstimator
    from adpstack.lane.classical import ClassicalLaneDetector
    from adpstack.pipeline import PerceptionPipeline
    from adpstack.tracking.bytetrack_wrapper import ByteTracker

    depth_cfg = load_config("configs/depth.yaml")
    lane_cfg = load_config("configs/lane.yaml")
    det = Detector(args.weights, imgsz=args.imgsz, device=args.device)
    depth = None
    if not args.no_depth:
        from adpstack.depth.midas_depth import MiDaSDepth

        depth = MiDaSDepth(depth_cfg["model"], depth_cfg["input_size"], args.device, args.midas_onnx)
    return PerceptionPipeline(det, ByteTracker(**load_config("configs/tracker.yaml")),
                              DistanceEstimator(depth_cfg), ClassicalLaneDetector(lane_cfg), depth,
                              depth_every=args.depth_every)


def run(args: argparse.Namespace) -> None:
    from adpstack.viz.overlay import draw_hud, draw_lane, draw_tracks

    pipe = build_pipeline(args)
    cap = cv2.VideoCapture(args.source)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and n >= args.max_frames):
            break
        res = pipe.process(frame, n / fps_in)
        vis = draw_lane(frame, res.lane.mask) if res.lane is not None else frame
        vis = draw_tracks(vis, res.objects)
        total = sum(res.timings_ms.values())
        vis = draw_hud(vis, int(res.risk.level), lane_offset_m=res.lane.offset_m if res.lane else None,
                       curvature_m=res.lane.curvature_m if res.lane else None, fps=1000 / total if total else None)
        if writer is None:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps_in,
                                     (vis.shape[1], vis.shape[0]))
        writer.write(vis)
        n += 1
    if writer:
        writer.release()
    log.info("wrote %s (%d frames); per-stage ms: %s", args.out, n, pipe.timer.summary())


def main() -> None:
    ap = argparse.ArgumentParser(prog="adp")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the full pipeline on a video")
    r.add_argument("--source", required=True)
    r.add_argument("--weights", required=True)
    r.add_argument("--out", default="outputs/demo.mp4")
    r.add_argument("--imgsz", type=int, default=640)
    r.add_argument("--device", default="cpu")
    r.add_argument("--no-depth", action="store_true")
    r.add_argument("--midas-onnx", default=None)
    r.add_argument("--depth-every", type=int, default=1)
    r.add_argument("--max-frames", type=int, default=0)
    r.set_defaults(fn=run)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
