# Results

> Rule for this file: **only measured numbers.** Rows marked `TODO` need a GPU run; the command to produce each is given.
> Hardware for every table must be stated. Fill in from `docs/figures/*.json` and W&B.

## 1. Verified in this repository (CPU, reproducible)
| Check | Result | Command |
|---|---|---|
| Unit tests | 30 / 30 passing (data conversion, EDA, ByteTrack, MOT metrics, lanes, depth calibration, TTC, risk, control, decoder, pipeline) | `pytest` |
| Closed-loop AEB logic, stationary obstacle at 60 m, ego 14 m/s, 20 noisy runs | no AEB: 100 % collisions, 7.8 m/s mean impact. **AEB on: 0 % collisions, min gap 4.7 m** | `python scripts/closed_loop_sim.py` |
| Closed-loop, slow lead car (5 m/s) at 45 m | no AEB: 100 % collisions, 7.9 m/s impact. **AEB on: 0 % collisions, min gap 10.3 m** | same |

Note: the closed-loop numbers come from a kinematic simulation (pinhole distance + 1.5 px noise, 0.1 s actuation latency,
8 m/s^2 max braking). They validate the control/TTC logic only, not perception accuracy and not CARLA behaviour.
Figure: `docs/figures/closed_loop_sim.png`. Preview frame: `docs/figures/synthetic_demo_frame.jpg`.

## 2. Dataset (BDD100K 10K) - TODO after `eda`
| Item | Value |
|---|---|
| Frames / boxes | TODO |
| Class counts (car ... traffic sign) | TODO (see `docs/figures/eda_summary.json`) |
| Imbalance ratio (max/min class) | TODO |
| Small objects (<32^2 px) | TODO |
| Night / rainy share | TODO |

## 3. Detection (Colab T4) - TODO
| Model | imgsz | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|---|
| YOLO11n | 640 | TODO | TODO | TODO | TODO |
| YOLOv8s | 640 | TODO | TODO | TODO | TODO |

Failure slices (`detection/evaluate.py`): night, rainy, small-objects - TODO. Per-class AP: TODO.

## 4. Tracking - TODO
| Detector | Tracker | MOTA | IDF1 | ID switches |
|---|---|---|---|---|
| YOLO11n | ByteTrack (default) | TODO | TODO | TODO |
| YOLO11n | ByteTrack (buffer 60) | TODO | TODO | TODO |

## 5. Lane detection - TODO
| Method | IoU (ego lane) | F1 | FPS (CPU) | FPS (T4) |
|---|---|---|---|---|
| Classical (OpenCV) | TODO | TODO | TODO | n/a |
| U-Net + MobileNetV2 | TODO | TODO | TODO | TODO |

Qualitative failure analysis (`docs/figures/lane_qualitative.png`): write 4-5 sentences on night, rain glare, worn paint, curves, occlusion.

## 6. Depth / distance - TODO (needs CARLA depth pairs)
| Method | MAE (m) | RMSE (m) | AbsRel | delta<1.25 |
|---|---|---|---|---|
| MiDaS_small + scale/shift fit | TODO | TODO | TODO | TODO |
| Pinhole + class height prior | TODO | TODO | TODO | TODO |
| Fused (50/50) | TODO | TODO | TODO | TODO |

## 7. ONNX / TensorRT benchmark - TODO (state the GPU)
Produced by `python -m adpstack.export.benchmark ...` (500 iterations, 50 warm-up).
| Backend | mean ms | p95 ms | FPS | speed-up |
|---|---|---|---|---|
| PyTorch FP32 (CPU) | TODO | TODO | TODO | 1.0x |
| ONNX Runtime (CPU) | TODO | TODO | TODO | TODO |
| PyTorch FP32 (T4) | TODO | TODO | TODO | TODO |
| ONNX Runtime CUDA (T4) | TODO | TODO | TODO | TODO |
| TensorRT FP16 (T4) | TODO | TODO | TODO | TODO |
| TensorRT INT8 (T4) | TODO | TODO | TODO | TODO |

INT8 mAP50-95 drop vs FP32: TODO.

## 8. CARLA scenarios (cloud VM) - TODO (`outputs/scenarios/*.json`)
| Scenario | Collisions | Min distance (m) | Reaction time (s) | E2E latency mean / p95 (ms) |
|---|---|---|---|---|
| pedestrian_crossing | TODO | TODO | TODO | TODO |
| lead_braking | TODO | TODO | TODO | TODO |
| cut_in | TODO | TODO | TODO | TODO |
| night_rain | TODO | TODO | TODO | TODO |

## 9. Honest limitations
- TensorRT and CARLA run on a cloud NVIDIA T4 because the development laptop has only an AMD integrated GPU.
- MiDaS is monocular: scale is calibrated on CARLA; transfer to real footage is approximate.
- Controller is a research prototype (no map, no planning), not a safety system.
