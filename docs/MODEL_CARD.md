# Model card: Autonomous Driving Perception Stack

## Intended use

This repository is a research and educational prototype for experimenting with object detection, tracking, lane segmentation, monocular depth, and perception-to-control components in offline datasets and simulated scenarios.

## Data

The planned detection workflow uses the BDD100K 10K image subset and its detection labels. Lane segmentation may use BDD100K drivable-area masks. CARLA-generated data is used for simulation experiments. Dataset versions, splits, and preprocessing must be recorded with each run.

## Metrics

Measured YOLO11n detection and TensorRT benchmark results, along with unrun tracking, lane, depth, and CARLA evaluations, are reported in [results.md](./results.md). The real-video examples are qualitative and do not establish tracking, lane, or distance accuracy. Do not treat smoke-test or simulation results as model-quality metrics.

## Limitations and risks

- Performance depends on dataset coverage, labeling, preprocessing, hardware, and configuration; results may not generalize to other regions, weather, cameras, or traffic conditions.
- Monocular depth estimates have uncertain absolute scale and require calibration.
- Simulation results do not establish real-world performance or safety.
- The repository does not provide a complete autonomous-driving system, safety case, or operational monitoring.

## Safety statement

This is a research prototype and is **not for use in real vehicles or safety-critical decisions**. Do not use its outputs to control a vehicle or to make decisions affecting road users.
