# TensorRT benchmark on Kaggle Tesla T4

YOLO11n `best.pt`, TensorRT 11.5.0.63, Ultralytics 8.4.215.

End-to-end Ultralytics `predict()` latency including pre/postprocess on a 1280x720 image,
batch 1, 300 iterations after 30 warm-up iterations. Accuracy on 2,000 val images.

| Backend | mean ms | p50 | p95 | p99 | FPS | speed-up | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PyTorch FP32 (CUDA) | 9.96 | 9.85 | 10.44 | 11.41 | 100.4 | 1.00x | 0.434 | 0.234 |
| TensorRT FP16 | 4.32 | 4.17 | 4.88 | 5.68 | 231.6 | 2.31x | 0.433 | 0.232 |
| TensorRT INT8 | 4.42 | 4.19 | 5.85 | 5.92 | 226.0 | 2.25x | 0.414 | 0.221 |
