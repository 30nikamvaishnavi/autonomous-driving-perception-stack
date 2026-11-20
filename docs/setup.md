# Setup guide

## 0. What runs where
| Task | Where | Why |
|---|---|---|
| Coding, unit tests, CPU inference, lane/tracking/fusion on videos | Windows laptop (CPU) | no GPU needed |
| YOLO / U-Net training, ONNX + TensorRT | Colab or Kaggle T4 | TensorRT needs an NVIDIA GPU |
| CARLA + ROS2 + demo video | Ubuntu 22.04 GPU VM (T4, e.g. AWS g4dn.xlarge spot) | CARLA needs ~6+ GB NVIDIA VRAM |

## 1. Laptop (Windows 11)
```powershell
git clone https://github.com/30nikamvaishnavi/autonomous-driving-perception-stack.git; cd autonomous-driving-perception-stack
py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"            # lightweight: numpy, opencv, scipy
pytest                             # CPU tests, no data needed
python scripts/synthetic_demo.py   # model-free demo -> outputs/synthetic_demo.mp4
pip install -e ".[ml,export]"      # PyTorch (CPU), ultralytics, ONNX (optional locally)
pip install onnxruntime-directml   # optional: use the AMD iGPU for the ONNX benchmark row
```

Online W&B tracking was not completed because authentication failed on a hosted W&B instance. Training metrics are available in `docs/figures/results.csv` and `docs/figures/training_curves.png`.

## 2. Data (Kaggle BDD100K subset)
The private Kaggle dataset `bdd100k-adpstack-subset` was made on Kaggle from the public BDD100K release
(10K Images and Detection 2020 `det_20` labels). The subset builder samples 10,000 available training
images and 2,000 validation images with seed 42, copies their images and labels, and normalizes the
category aliases `bike` to `bicycle` and `motor` to `motorcycle`.

To reproduce the subset from a BDD100K root that has the layout below:
```powershell
python scripts/make_bdd_subset.py --source data/bdd100k --out data/bdd100k-adpstack-subset
```

Attach the resulting folder as a Kaggle dataset named `bdd100k-adpstack-subset`. Enable a GPU
accelerator and Internet in the Kaggle notebook settings. Set the Kaggle secret `WANDB_API_KEY`
only if you want online W&B logging; the training notebook runs without it.

Expected input layout before building the subset:
```
data/bdd100k/images/{train,val}/*.jpg
data/bdd100k/labels/det_20/det_{train,val}.json
```
Convert + EDA:
```powershell
python -m adpstack.data.bdd100k_to_yolo --subset 10000
python -m adpstack.data.eda --labels data/bdd100k/labels/det_20/det_train.json --out docs/figures --wandb
```
The completed Kaggle run used YOLO11n at 640 pixels, batch 16, and 50 epochs on a T4. Total
training time was 4,571.65 seconds (about 76 minutes).
CARLA synthetic frames (second domain) come from `carla_recorder` on the VM (section 5).

## 3. Colab (training) - use `notebooks/02_yolo_train_colab.ipynb`
Upload the converted data to Drive, run the notebook, save `best.pt` to Drive.
U-Net lane model training was not run; no trained U-Net checkpoint or evaluation results are available.

## 4. TensorRT on Colab T4 - `notebooks/03_tensorrt_colab.ipynb`
Produces `docs/figures/benchmark.{md,json,png}` with PyTorch / ONNX Runtime / TRT FP16 / TRT INT8 latency + FPS.

## 5. GPU VM: CARLA 0.9.15 + ROS2 Humble
```bash
# Ubuntu 22.04 + NVIDIA driver + docker + nvidia-container-toolkit (use the AWS "Deep Learning Base GPU AMI (Ubuntu 22.04)")
docker run -d --name carla --gpus all --net=host carlasim/carla:0.9.15 ./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Low

# ROS2 Humble
sudo apt update && sudo apt install -y ros-humble-ros-base ros-humble-rqt-image-view python3-colcon-common-extensions python3-pip
git clone https://github.com/30nikamvaishnavi/autonomous-driving-perception-stack.git && cd autonomous-driving-perception-stack
pip3 install "numpy<2" opencv-python-headless scipy pyyaml matplotlib onnxruntime-gpu wandb carla==0.9.15
pip3 install -e . --no-deps
source /opt/ros/humble/setup.bash && cd ros2_ws && colcon build --symlink-install && cd ..
source ros2_ws/install/setup.bash

# record synthetic data
python3 -m adpstack.data.carla_recorder --weather ClearNoon && python3 -m adpstack.data.carla_recorder --weather HardRainNight

# run one scenario with views
ros2 launch adp_bringup full_stack.launch.py scenario:=pedestrian_crossing weights:=models/best.onnx
ros2 run rqt_image_view rqt_image_view /perception/overlay     # via X forwarding / NICE DCV
# all scenarios + metrics JSON
scripts/run_scenarios.sh models/best.onnx
```
Record the demo with OBS (NICE DCV / Remote desktop) or `ffmpeg -f x11grab`. **Stop the VM when finished.**

## 6. Calibrating MiDaS -> metres
Record paired frames (`carla_recorder` stores `depth/*.npy`), run `MiDaSDepth` on the RGB, call
`depth.calibrate_scale.fit_over_dataset`, then put the resulting scale/shift in `configs/depth.yaml` (or fusion_node params).

## 7. Troubleshooting
- `import carla` fails -> Python must match the wheel (3.10 for ROS Humble on 22.04).
- Black camera frames in rain/night -> raise exposure (`exposure_mode`) in the camera blueprint.
- ONNX CUDA provider missing -> `pip uninstall onnxruntime && pip install onnxruntime-gpu`.
