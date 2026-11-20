# Experiment log

Record only completed runs with measured results in the table. Planned work is listed separately;
do not estimate or copy metrics from smoke tests.

| Run name | W&B link | Config | mAP50 | mAP50-95 |
|---|---|---|---:|---:|
| YOLO11n, 50 epochs, Kaggle Tesla T4 | Not recorded | imgsz 640, batch 16; 10,000 train / 2,000 val images | 0.434 | 0.234 |
| YOLO11n TensorRT benchmark, Kaggle Tesla T4 | Not recorded | YOLO11n `best.pt`, TensorRT 11.5.0.63, Ultralytics 8.4.215; 2,000 val images | 0.434 (PyTorch FP32), 0.433 (FP16), 0.414 (INT8) | 0.234 (FP32), 0.232 (FP16), 0.221 (INT8) |

| Backend | mean ms | p50 | p95 | p99 | FPS | speed-up |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch FP32 (CUDA) | 9.96 | 9.85 | 10.44 | 11.41 | 100.4 | 1.00x |
| TensorRT FP16 | 4.32 | 4.17 | 4.88 | 5.68 | 231.6 | 2.31x |
| TensorRT INT8 | 4.42 | 4.19 | 5.85 | 5.92 | 226.0 | 2.25x |

Method: end-to-end Ultralytics `predict()` latency with batch 1 on a 1280x720 image,
300 iterations after 30 warm-up iterations. Ultralytics' inference-only times were
PyTorch 9.5 ms, FP16 2.9 ms, and INT8 3.0 ms.

ONNX size: 10.2 MB. FP16 engine: 55 MB, 157 s build time. INT8 engine: 43 MB, 198 s build time,
calibrated on 300 val images from the same pool used for evaluation.

INT8 was not faster than FP16 here and lost 0.020 mAP50 versus FP32.
The repository exporter is incompatible with TensorRT 11.5 (`BuilderFlag has no attribute FP16`);
the Kaggle run used the Ultralytics exporter.

Precision: **0.569**; recall: **0.409**.

| Class | AP50-95 |
|---|---:|
| Car | 0.415 |
| Truck | 0.369 |
| Bus | 0.341 |
| Motorcycle | 0.109 |
| Bicycle | 0.142 |
| Person | 0.208 |
| Rider | 0.117 |
| Traffic light | 0.163 |
| Traffic sign | 0.247 |

## Planned experiment (not run)

Train YOLO11n for 50 epochs at imgsz 640 and batch 16, keeping the other training settings
unchanged, but sample each training image containing motorcycle, rider, or bicycle objects three
times as often as under the current image sampling scheme. **Not run:** no resampling comparison or
runtime/cost measurement is available.
