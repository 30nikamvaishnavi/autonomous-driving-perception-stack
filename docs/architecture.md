# Architecture

## System overview
```mermaid
flowchart LR
  subgraph CARLA["CARLA 0.9.15 (GPU VM)"]
    EGO[Ego vehicle] --> RGB[RGB camera]
    EGO --> DEP[Depth camera GT]
  end
  RGB -->|/camera/image_raw| DET[detector_node<br/>YOLO ONNX/TensorRT]
  RGB --> LANE[lane_node<br/>classical or U-Net]
  RGB --> MID[depth_node<br/>MiDaS]
  DET -->|/perception/detections| TRK[tracker_node<br/>ByteTrack]
  TRK -->|/perception/tracks| FUS[fusion_node<br/>distance, TTC, risk]
  MID -->|/perception/depth_rel| FUS
  DEP -.->|/camera/depth, optional GT| FUS
  FUS -->|/perception/risk, /perception/objects| CTL[controller_node<br/>AEB + ACC + LKA]
  LANE -->|/perception/lanes| CTL
  CTL -->|/vehicle/control_cmd| EGO
  FUS -->|/perception/overlay| VIZ[rqt_image_view / demo video]
  CTL --> MET[metrics_node -> JSON + W&B]
```

## ROS2 graph (topics)
```mermaid
flowchart TD
  bridge((carla_bridge)) -- Image --> detector((detector_node)) -- DetectionArray --> tracker((tracker_node))
  tracker -- TrackedObjectArray --> fusion((fusion_node))
  bridge -- Image --> depth((depth_node)) -- Image 32FC1 --> fusion
  bridge -- Image --> lane((lane_node)) -- LaneInfo --> controller((controller_node))
  fusion -- Risk --> controller
  fusion -- TrackedObjectArray --> controller
  bridge -- Float32 speed --> controller
  controller -- VehicleControl --> bridge
  bridge -- Bool collision / Float32 gt_distance --> metrics((metrics_node))
```

## Per-frame data flow
```mermaid
sequenceDiagram
  participant C as Camera
  participant D as Detector
  participant T as ByteTrack
  participant M as MiDaS
  participant F as Fusion
  participant K as Controller
  C->>D: BGR frame
  C->>M: BGR frame (every N)
  D->>T: boxes, conf, cls
  T->>F: tracks with persistent IDs
  M->>F: relative inverse depth
  F->>F: metric depth (scale/shift) + pinhole prior -> distance
  F->>F: Kalman -> closing speed -> TTC -> risk
  F->>K: Risk(level, id, dist, TTC)
  K->>C: throttle / brake / steer (AEB overrides)
```

## Deployment
```mermaid
flowchart LR
  L[Windows laptop<br/>VS Code + Copilot<br/>CPU dev + tests] -->|git push| GH[GitHub + CI]
  GH --> CO[Colab/Kaggle T4<br/>train YOLO + seg<br/>ONNX + TensorRT bench]
  CO -->|best.pt / .onnx| VM[Ubuntu 22.04 T4 VM<br/>CARLA + ROS2 Humble + Docker]
  VM --> W[(Weights & Biases)]
  CO --> W
```

## Design decisions
- **Backend-agnostic detector** (`Detector`): same code path for `.pt`, `.onnx` (CPU / DirectML / CUDA / TensorRT EP).
- **Own ByteTrack + MOT metrics** (numpy/scipy): unit-testable on CPU, no heavy dependencies.
- **Distance = fusion of two independent estimators** (MiDaS calibrated by least squares vs. CARLA depth; pinhole + class height prior).
- **TTC from a Kalman-filtered distance**, never from raw frame-to-frame differences (noise).
- **AEB overrides ACC overrides cruise**; lane keeping uses the Stanley law on the lane offset.
- **Everything is configured in `configs/*.yaml`** with dotted overrides (`--set epochs=5`).
