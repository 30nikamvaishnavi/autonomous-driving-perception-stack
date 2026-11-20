# Individual reflection

**Name:** Vaishnavi Nikam  
**Project:** Autonomous Driving Perception Stack  
**Date:** Not provided

## What I built

- Implemented BDD100K label conversion, subset preparation, and EDA. The converter normalizes `bike` to `bicycle` and `motor` to `motorcycle`; tests cover both aliases.
- Built perception components for YOLO11n detection, ByteTrack tracking, classical lane detection, depth/distance estimation, TTC/risk fusion, and control.
- Added CARLA bridge and ROS2 node code. The CARLA + ROS2 system was not run.
- Produced a real-video pipeline demo, a kinematic braking simulation, and training/benchmark result artifacts.

## What I learned

- Technical: [YOUR WORDS: describe what you learned]
- Process, testing, reproducibility, and experiment tracking: [YOUR WORDS: describe what you learned]

## What went wrong and how I fixed it

1. BDD100K labels include `bike` and `motor` aliases. The converter maps them to the project class names `bicycle` and `motorcycle`, and unit tests verify the conversion.
2. The repository TensorRT exporter is incompatible with TensorRT 11.5 (`BuilderFlag has no attribute FP16`). The Kaggle benchmark used the Ultralytics exporter instead.
3. Online W&B tracking did not complete because authentication failed on a hosted W&B instance. Metrics remain available in [`figures/results.csv`](figures/results.csv) and [`figures/training_curves.png`](figures/training_curves.png); no online W&B run is claimed.

## Results

- YOLO11n training ran for 50 epochs on a Kaggle Tesla T4 and took 4,571.65 s. Epoch 39 had validation mAP50 0.43376 and mAP50-95 0.23392; epoch 50 had 0.4295 and 0.23294.
- Separate `best.pt` evaluation reported mAP50 0.43375 and mAP50-95 0.23445.
- TensorRT FP16 measured 231.6 FPS compared with 100.4 FPS for PyTorch FP32 in the documented benchmark setup.
- Tracking, lane, and MiDaS distance accuracy were not measured; U-Net training and CARLA + ROS2 execution were not run.

## What I would do next

[YOUR WORDS: describe your intended next steps]

## What I felt about the project

[YOUR WORDS: describe your experience in your own words]
