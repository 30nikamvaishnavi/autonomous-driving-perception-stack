# Submission notes

The Autonomous Driving Perception Stack is a research prototype for detection, tracking, lane processing, distance/TTC/risk fusion, and control experiments. The repository includes data-preparation tools, YOLO11n training and evaluation, a video pipeline, and CARLA/ROS2 integration code. The CARLA and ROS2 components have not been run together.

The BDD100K workflow created a 10,000-image training subset and a 2,000-image validation subset; the dataset itself is not included. YOLO11n training ran for 50 epochs on a Kaggle Tesla T4 and took 4,571.65 seconds (about 76 minutes). Epoch 39 had the highest validation mAP50-95 in the training CSV: mAP50 0.43376 and mAP50-95 0.23392. The last epoch had 0.4295 and 0.23294. Separate `best.pt` evaluation reported mAP50 0.43375 and mAP50-95 0.23445. TensorRT FP16 measured 231.6 FPS versus 100.4 FPS for PyTorch FP32 in the documented benchmark setup. The training curves and results CSV are included.

Tracking accuracy, lane accuracy, and MiDaS distance error are not measured. The U-Net lane model was not trained. CARLA scenarios were not run, and the kinematic braking simulation is not CARLA. Online W&B tracking was not completed because authentication failed on a hosted W&B instance; no online run is claimed.

For CPU-only checks on Windows, create and activate a virtual environment, install the development dependencies with `python -m pip install -e ".[dev]"`, then run `python -m pytest` and `python scripts/closed_loop_sim.py`. The simulation checks control logic, not perception accuracy or CARLA behavior. See the [README](../README.md) for setup details and the [architecture](./architecture.md) for the system design.

Live results page: https://30nikamvaishnavi.github.io/autonomous-driving-perception-stack/
