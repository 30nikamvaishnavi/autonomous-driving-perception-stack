# API reference

## ROS2 topics
| Topic | Type | Publisher | Notes |
|---|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` (bgr8) | carla_bridge | 1280x720 @ 20 Hz |
| `/camera/depth` | `sensor_msgs/Image` (32FC1) | carla_bridge | ground-truth metres |
| `/camera/info` | `sensor_msgs/CameraInfo` | carla_bridge | pinhole K |
| `/vehicle/speed` | `std_msgs/Float32` | carla_bridge | m/s |
| `/vehicle/collision` | `std_msgs/Bool` | carla_bridge | per collision event |
| `/scenario/gt_distance` | `std_msgs/Float32` | carla_bridge | ego -> scenario target (metrics only) |
| `/perception/detections` | `adp_msgs/DetectionArray` | detector_node | |
| `/perception/tracks` | `adp_msgs/TrackedObjectArray` | tracker_node | no distance yet |
| `/perception/lanes` | `adp_msgs/LaneInfo` | lane_node | curvature, offset, fits |
| `/perception/lane_overlay` | `sensor_msgs/Image` | lane_node | |
| `/perception/depth_rel` | `sensor_msgs/Image` (32FC1) | depth_node | MiDaS inverse depth |
| `/perception/objects` | `adp_msgs/TrackedObjectArray` | fusion_node | with distance/TTC |
| `/perception/risk` | `adp_msgs/Risk` | fusion_node | stamp = source image stamp |
| `/perception/overlay` | `sensor_msgs/Image` | fusion_node | demo visual |
| `/vehicle/control_cmd` | `adp_msgs/VehicleControl` | controller_node | |

Messages: see `ros2_ws/src/adp_msgs/msg/*.msg`.

## Node parameters
| Node | Parameter | Default |
|---|---|---|
| bridge | scenario (`free_drive`, `pedestrian_crossing`, `lead_braking`, `cut_in`, `night_rain`), town, fps, host, seed | free_drive |
| detector | weights, imgsz, conf, providers | best.onnx, 640, 0.25 |
| lane | backend (`classical`/`segmentation`), seg_weights | classical |
| depth | onnx_path, every_n | "", 2 |
| fusion | use_gt_depth, scale, shift | false, 1, 0 |
| controller | config, rate_hz | configs/control.yaml, 20 |
| metrics | scenario, duration_s, out_dir, wandb | unnamed, 30 |

## Python API (`src/adpstack`)
```python
from adpstack.detection.infer import Detector
from adpstack.tracking.bytetrack_wrapper import ByteTracker
from adpstack.lane.classical import ClassicalLaneDetector
from adpstack.pipeline import PerceptionPipeline

det = Detector("best.onnx")                         # or best.pt
pipe = PerceptionPipeline(det, ByteTracker(), DistanceEstimator(cfg), ClassicalLaneDetector(lane_cfg), MiDaSDepth())
res = pipe.process(bgr_frame, t_seconds)            # FrameResult: objects, lane, risk, timings_ms
```

## CLI
| Command | Purpose |
|---|---|
| `python -m adpstack.data.bdd100k_to_yolo --subset 10000` | BDD100K JSON -> YOLO txt + dataset.yaml |
| `python -m adpstack.data.eda --labels ... --wandb` | class balance, sizes, attribute splits |
| `python -m adpstack.detection.train --set epochs=50 model=yolo11n.pt` | fine-tune (W&B logged) |
| `python -m adpstack.detection.evaluate --weights best.pt ...` | mAP, per-class AP, night/rain/small slices |
| `python -m adpstack.lane.segmentation train --images ... --masks ...` | train U-Net |
| `python -m adpstack.lane.compare --images ... --masks ... --seg-weights ...` | classical vs DL |
| `python -m adpstack.export.to_onnx / to_tensorrt / benchmark` | export + latency tables |
| `python -m adpstack.data.carla_recorder --weather HardRainNight` | synthetic dataset |
| `adp run --source video.mp4 --weights best.onnx --out demo.mp4` | full pipeline on a video |
| `python scripts/closed_loop_sim.py` / `scripts/synthetic_demo.py` | model-free checks |

## Config schema (`configs/`)
`yolo.yaml` (model, data, epochs, imgsz, augment.*, wandb.*), `tracker.yaml` (ByteTrack thresholds), `lane.yaml`
(ROI trapezoid, BEV size, metres/pixel, sliding window, seg.*), `depth.yaml` (MiDaS, camera intrinsics,
class heights, scale/shift), `control.yaml` (aeb, acc, lka gains), `carla.yaml` (server, camera, weather, recording).
