# Autonomous Driving Perception Stack

[![CI](https://github.com/30nikamvaishnavi/autonomous-driving-perception-stack/actions/workflows/ci.yml/badge.svg)](https://github.com/30nikamvaishnavi/autonomous-driving-perception-stack/actions)

This repository is an end-to-end research implementation for an autonomous-driving perception stack:
BDD100K subset preparation and EDA, YOLO11n detection, ByteTrack tracking, classical lane detection,
distance/TTC/risk fusion, AEB/ACC/lane control, ONNX/TensorRT export, CARLA scenarios, and ROS2 nodes.
The project is organized so CPU-only components can be tested on a laptop, while training and TensorRT
benchmarks run on Kaggle/Colab GPU sessions.

## Status

| Component | State |
|---|---|
| BDD100K 10,000-image train / 2,000-image validation subset and EDA | **Done** |
| YOLO11n training | **Done** - mAP50 **0.434**, mAP50-95 **0.234** |
| TensorRT FP16/INT8 benchmark on Kaggle Tesla T4 | **Done** - FP16 231.6 FPS, INT8 226.0 FPS |
| ByteTrack tracking | **Code and unit tests done**; no real tracking-data evaluation yet |
| Classical lane detection | **Code and tests done**; no real-image numbers yet |
| U-Net lane model and comparison | **Not trained** |
| MiDaS distance and TTC/risk fusion | **Code and calibration functions done**; no real error numbers yet |
| CARLA + ROS2 implementation | **Implemented but not run** |
| W&B logging | **Wired**; Kaggle-run delivery to a W&B account still needs confirmation |
| Obstacle-reaction demo video | **Not done**; requires CARLA |
| 10-minute presentation and reflection | **Outline and template only** |

## Quickstart (no GPU needed)

On Windows PowerShell:

```powershell
git clone https://github.com/30nikamvaishnavi/autonomous-driving-perception-stack.git
cd autonomous-driving-perception-stack
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
python scripts/closed_loop_sim.py
python scripts/synthetic_demo.py
```

Run these commands:

- `python -m pytest` - runs the CPU-only test suite.
- `python scripts/closed_loop_sim.py` - runs the kinematic AEB/ACC check and writes
  `docs/figures/closed_loop_sim.json` and `docs/figures/closed_loop_sim.png`.
- `python scripts/synthetic_demo.py` - generates a model-free HUD/BEV preview video at
  `outputs/synthetic_demo.mp4`.

## Run on a GPU

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/30nikamvaishnavi/autonomous-driving-perception-stack/blob/main/notebooks/00_gpu_smoke_test.ipynb)

- [YOLO training notebook](notebooks/04_kaggle_yolo_train.ipynb) - Kaggle GPU; attach the private
  `bdd100k-adpstack-subset` dataset and a BDD100K-license-accepted account.
- [TensorRT Kaggle notebook](notebooks/05_tensorrt_kaggle.ipynb) - Kaggle Tesla T4; needs `best.pt`,
  the YOLO `dataset.yaml`, TensorRT, and Ultralytics.
- [Pipeline video notebook](notebooks/06_pipeline_video_kaggle.ipynb) - Kaggle GPU; needs trained
  `best.pt` plus a short BDD100K or dashcam MP4.

Other setup and VM instructions are in [`docs/setup.md`](docs/setup.md).

## Data

BDD100K is not included in this repository. The script
[`scripts/make_bdd_subset.py`](scripts/make_bdd_subset.py) expects a BDD100K root with:

```text
data/bdd100k/images/{train,val}/*.jpg
data/bdd100k/labels/det_20/det_{train,val}.json
```

Build the reproducible subset with seed 42:

```powershell
python scripts/make_bdd_subset.py --source data/bdd100k --out data/bdd100k-adpstack-subset
```

The script builds 10,000 training images and 2,000 validation images and writes copied images plus
the selected detection JSON files. Users must obtain BDD100K and accept the BDD100K license terms
before downloading or using the dataset.

## Results

### Detection (Kaggle Tesla T4)

Measured setup: YOLO11n, image size 640, batch 16, 50 epochs; 10,000 training and 2,000
validation images.

| Model | imgsz | Epochs | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLO11n | 640 | 50 | 0.434 | 0.234 |

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

### ONNX / TensorRT benchmark

Measured on Kaggle with a Tesla T4, YOLO11n `best.pt`, TensorRT 11.5.0.63, and Ultralytics 8.4.215.
Latency is end-to-end Ultralytics `predict()` including pre/postprocessing on a 1280x720 image,
batch 1, 300 iterations after 30 warm-up iterations. Accuracy was measured on 2,000 val images.

| Backend | mean ms | p50 | p95 | p99 | FPS | speed-up | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PyTorch FP32 (CUDA) | 9.96 | 9.85 | 10.44 | 11.41 | 100.4 | 1.00x | 0.434 | 0.234 |
| TensorRT FP16 | 4.32 | 4.17 | 4.88 | 5.68 | 231.6 | 2.31x | 0.433 | 0.232 |
| TensorRT INT8 | 4.42 | 4.19 | 5.85 | 5.92 | 226.0 | 2.25x | 0.414 | 0.221 |

Ultralytics' inference-only times were 9.5 ms for PyTorch, 2.9 ms for FP16, and 3.0 ms for INT8.
The ONNX export was 10.2 MB; the FP16 engine was 55 MB and took 157 s to build; the INT8 engine
was 43 MB and took 198 s to build.

Limitations:

- The same 2,000 validation images were used to select `best.pt` and report detection metrics,
  so those results are slightly optimistic.
- INT8 calibration used 300 val images from the same 2,000-image pool used for evaluation.
- TensorRT INT8 was not faster than FP16 here and lost 0.020 mAP50 versus FP32.
- The repository exporter in [`src/adpstack/export/to_tensorrt.py`](src/adpstack/export/to_tensorrt.py)
  is incompatible with TensorRT 11.5 (`BuilderFlag has no attribute FP16`); the benchmark used
  the Ultralytics exporter.

Full results and ongoing TODO items are in [`docs/results.md`](docs/results.md).

## Mapping to the internship brief

| Item | Deliverable | Status |
|---|---|---|
| 1. BDD100K, labels, class balance | Reproducible 10K/2K subset and measured class counts | **Done** |
| 2. YOLO fine-tune | YOLO11n checkpoint and mAP50/mAP50-95 | **Done** - mAP50 0.434 |
| 2. ByteTrack tracking | ByteTrack code and unit tests | **Partial** - not evaluated on real tracking data |
| 3. Classical lane detection | Classical OpenCV lane code and tests | **Partial** - no real-image numbers |
| 3. Segmentation model and comparison | Segmentation implementation/training path | **Not done** - model not trained |
| 4. MiDaS distance | MiDaS and calibration functions | **Partial** - no real error numbers |
| 5. CARLA + ROS2 | CARLA bridge, perception/control nodes, scenarios | **Partial** - code written, never run |
| 6. ONNX and TensorRT benchmark | ONNX export and FP16/INT8 benchmark | **Done** - 2.31x FP16 speed-up |
| 7. W&B logging | Notebook/logging integration | **Partial** - confirm Kaggle runs appear in your W&B account |
| 7. Obstacle-reaction demo video | Recorded CARLA reaction demo | **Not done** - requires CARLA |
| Presentation, reflection, repository | Architecture docs, tests, README, 10-minute outline, reflection template | **Partial** - outline/template only; final presentation/reflection remain |

## Repository map

```text
configs/        Project and module configuration
src/adpstack/   Data, detection, tracking, lane, depth, fusion, control, export, viz, pipeline
ros2_ws/src/    CARLA bridge, detector/perception, control, and ROS2 launch packages
notebooks/      Kaggle/Colab training and benchmark notebooks
scripts/        Subset creation, closed-loop simulation, synthetic demo, scenario runner
docker/         ROS2 and training containers
docs/           Architecture, setup, experiments, results, and benchmark figures
tests/          CPU-only pytest suite
```

## License

MIT
