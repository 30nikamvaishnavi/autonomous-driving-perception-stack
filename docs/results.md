# Results

> Rule for this file: **only measured numbers.** Unrun evaluations are labeled `Not run`.
> Hardware is stated for measured GPU results. Online W&B tracking was not completed because authentication failed on a hosted W&B instance; metrics are in `docs/figures/results.csv` and `docs/figures/training_curves.png`.

## 1. Verified in this repository (CPU, reproducible)
| Check | Result | Command |
|---|---|---|
| Unit tests | 49 / 49 passing (data conversion, EDA, ByteTrack, MOT metrics, lanes, depth calibration, TTC, risk, control, decoder, pipeline) | `.\.venv\Scripts\python.exe -m pytest` |
| Closed-loop AEB logic, stationary obstacle at 60 m, ego 14 m/s, 20 noisy runs | no AEB: 100 % collisions, 7.8 m/s mean impact. **AEB on: 0 % collisions, min gap 4.7 m** | `python scripts/closed_loop_sim.py` |
| Closed-loop, slow lead car (5 m/s) at 45 m | no AEB: 100 % collisions, 7.9 m/s impact. **AEB on: 0 % collisions, min gap 10.3 m** | same |

Note: the closed-loop numbers come from a kinematic simulation (pinhole distance + 1.5 px noise, 0.1 s actuation latency,
8 m/s^2 max braking). They validate the control/TTC logic only, not perception accuracy and not CARLA behaviour.
Figure: `docs/figures/closed_loop_sim.png`. Preview frame: `docs/figures/synthetic_demo_frame.jpg`.

## 2. Dataset (BDD100K Kaggle subset)
Measured box counts for the 10,000-image training and 2,000-image validation subsets:

| Class | Train boxes | Validation boxes |
|---|---:|---:|
| Car | 103,386 | 20,375 |
| Traffic sign | 34,587 | 6,989 |
| Traffic light | 26,942 | 5,466 |
| Person | 13,574 | 2,500 |
| Truck | 4,100 | 866 |
| Bus | 1,595 | 311 |
| Bicycle | 1,012 | 178 |
| Rider | 627 | 136 |
| Motorcycle | 427 | 100 |

Train car-to-motorcycle imbalance ratio: **242.12:1** (103,386 / 427). Mitigation to evaluate
in a later training comparison: class weights, mosaic/mixup augmentation, and minority-class
oversampling. These are proposed experiments, not measured improvements.

| Other dataset statistic | Value |
|---|---|
| Small objects (<32^2 px) | Not run |
| Night / rainy share | Not run |

Small-object and night/rainy share analyses were not run because BDD100K is not included in this repository.

## 3. YOLO11n training (Kaggle Tesla T4)

The 50-epoch run took **4,571.65 s** (about 76 minutes). The highest validation mAP50-95 in `docs/figures/results.csv` was at epoch 39. The final row is epoch 50.

| Epoch | mAP50 | mAP50-95 |
|---:|---:|---:|
| 39 (best validation mAP50-95) | 0.43376 | 0.23392 |
| 50 (last) | 0.42950 | 0.23294 |

Training losses continued to fall while validation mAP stayed roughly flat after about epoch 39. This is a **possible sign of overfitting, not a proven conclusion**.

## 4. Detection (Kaggle Tesla T4)
Measured setup: YOLO11n, imgsz 640, batch 16, 50 epochs; 10,000 training and 2,000
validation images from the private BDD100K subset.

The separate `best.pt` evaluation in `docs/figures/detection_eval.json` reports mAP50 **0.43375369780511985**, mAP50-95 **0.23445463247137935**, precision **0.5685970913216417**, and recall **0.4086158581282986**. These checkpoint-evaluation metrics differ from the epoch-39 validation values above.

Precision: **0.569**; recall: **0.409**.

| Class | Train boxes | AP50-95 |
|---|---:|---:|
| Car | 103,386 | 0.415 |
| Truck | 4,100 | 0.369 |
| Bus | 1,595 | 0.341 |
| Motorcycle | 427 | 0.109 |
| Bicycle | 1,012 | 0.142 |
| Person | 13,574 | 0.208 |
| Rider | 627 | 0.117 |
| Traffic light | 26,942 | 0.163 |
| Traffic sign | 34,587 | 0.247 |

Failure analysis — **Measured:** motorcycle, rider, bicycle, and traffic light have the weakest
per-class AP; motorcycle, rider, and bicycle are the three rarest classes by training-box count
and also have the three lowest AP values. Truck and bus score relatively well despite low training
counts, while traffic light has a comparatively high count but weak AP. **Hypothesis to test:**
small traffic-light objects may contribute to its weak AP; object size was not measured in this
evaluation, so this is not an established cause.

Limitation: the same 2,000 validation images were used to select `best.pt` and report these
metrics, so the results are slightly optimistic. No separate test split was used.

## 5. Tracking
| Detector | Tracker | MOTA | IDF1 | ID switches |
|---|---|---|---|---|
| YOLO11n | ByteTrack (default) | Not run | Not run | Not run |
| YOLO11n | ByteTrack (buffer 60) | Not run | Not run | Not run |

Tracking accuracy evaluation was not run because no real tracking-data evaluation or ground-truth comparison was completed.

## 6. Lane detection
| Method | IoU (ego lane) | F1 | FPS (CPU) | FPS (T4) |
|---|---|---|---|---|
| Classical (OpenCV) | Not run | Not run | Not run | Not run |
| U-Net + MobileNetV2 | Not run | Not run | Not run | Not run |

Lane metrics were not run because no labeled lane-ground-truth evaluation was completed. U-Net training/evaluation was not run. The real-video examples show lane-overlay failures; see the screenshots in the README.

## 7. Depth / distance
| Method | MAE (m) | RMSE (m) | AbsRel | delta<1.25 |
|---|---|---|---|---|
| MiDaS_small + scale/shift fit | Not run | Not run | Not run | Not run |
| Pinhole + class height prior | Not run | Not run | Not run | Not run |
| Fused (50/50) | Not run | Not run | Not run | Not run |

Distance-error evaluation was not run because paired CARLA depth ground truth is unavailable; CARLA was not run.

## 8. ONNX / TensorRT benchmark
Measured on Kaggle with a Tesla T4, YOLO11n `best.pt`, TensorRT 11.5.0.63, and Ultralytics 8.4.215.
Latency is end-to-end Ultralytics `predict()` including pre/postprocessing on a 1280x720 image, batch 1,
with 300 iterations after 30 warm-up iterations. Accuracy was measured on 2,000 val images.

| Backend | mean ms | p50 | p95 | p99 | FPS | speed-up | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PyTorch FP32 (CUDA) | 9.96 | 9.85 | 10.44 | 11.41 | 100.4 | 1.00x | 0.434 | 0.234 |
| TensorRT FP16 | 4.32 | 4.17 | 4.88 | 5.68 | 231.6 | 2.31x | 0.433 | 0.232 |
| TensorRT INT8 | 4.42 | 4.19 | 5.85 | 5.92 | 226.0 | 2.25x | 0.414 | 0.221 |

Ultralytics' own inference-only times were 9.5 ms for PyTorch, 2.9 ms for FP16, and 3.0 ms for INT8.
ONNX export size was 10.2 MB. The FP16 engine was 55 MB and built in 157 s; the INT8 engine was 43 MB
and built in 198 s.

Notes: TensorRT INT8 was not faster than FP16 on this run and lost 0.020 mAP50 versus FP32.
INT8 calibration used 300 validation images from the same 2,000-image validation pool used for evaluation.
The repository exporter at `src/adpstack/export/to_tensorrt.py` is incompatible with TensorRT 11.5:
it fails with `BuilderFlag has no attribute FP16`. This is a known issue; the Kaggle run used the
Ultralytics exporter. Latency values are end-to-end and are not directly comparable with inference-only times.

## 9. Real-video pipeline timing

Measured on a Kaggle Tesla T4 with YOLO11n `best.pt`, `--no-depth`, and a maximum of 300 frames per
clip. Values are median per-stage latency in milliseconds; they exclude video decoding and drawing.

| Pexels video | Input size | Frames | Detect (ms) | Track (ms) | Lane (ms) | Fusion (ms) |
|---|---:|---:|---:|---:|---:|---:|
| 3602494 | 1920x1080 | 300 | 12.7 | 0.8 | 74.1 | 0.48 |
| 3978374 | 1920x1080 | 213 | 12.7 | 0.7 | 78.4 | 0.31 |
| 14966361 | 2560x1440 output | 300 | 12.9 | 0.5 | 135.6 | 0.05 |
| 4608285 | 3840x2160 | 300 | 13.5 | 0.9 | 258.4 | 0.46 |
| 5921059 | 3840x2160 | 300 | 13.5 | 1.3 | 255.9 | 0.34 |

The classical lane detector dominates the measured pipeline latency, and its latency grows
substantially with input resolution. Distances use the pinhole prior with camera intrinsics from
`configs/depth.yaml`, which assumes 1280x720; therefore, these distances are illustrative, not
calibrated. There is no ground truth for tracking, lanes, or distance on these clips, so no accuracy
claims are made for those outputs.

## 10. CARLA scenarios (cloud VM)
| Scenario | Collisions | Min distance (m) | Reaction time (s) | E2E latency mean / p95 (ms) |
|---|---|---|---|---|
| pedestrian_crossing | Not run | Not run | Not run | Not run |
| lead_braking | Not run | Not run | Not run | Not run |
| cut_in | Not run | Not run | Not run | Not run |
| night_rain | Not run | Not run | Not run | Not run |

CARLA scenario evaluation was not run because CARLA + ROS2 was not run.

## 11. Honest limitations
- TensorRT benchmarking ran on a Kaggle Tesla T4. CARLA was not run.
- MiDaS is monocular; distance-error evaluation and calibration against CARLA depth were not run.
- Controller is a research prototype (no map, no planning), not a safety system.
