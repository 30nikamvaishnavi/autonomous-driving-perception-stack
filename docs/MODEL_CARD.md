# Model card: Autonomous Driving Perception Stack

## Intended use

This repository is a research and educational prototype for experimenting with object detection, tracking, lane segmentation, monocular depth, and perception-to-control components in offline datasets and simulated scenarios.

## Data

The planned detection workflow uses the BDD100K 10K image subset and its detection labels. Lane segmentation may use BDD100K drivable-area masks. CARLA-generated data is used for simulation experiments. Dataset versions, splits, and preprocessing must be recorded with each run.

## Metrics

No trained-model performance is reported here. Add measured detection, tracking, lane, depth, and latency results to [results.md](./results.md) and [experiments.md](./experiments.md) after running the documented evaluations. Do not treat smoke-test results as model-quality metrics.

## Limitations and risks

- Performance depends on dataset coverage, labeling, preprocessing, hardware, and configuration; results may not generalize to other regions, weather, cameras, or traffic conditions.
- Monocular depth estimates have uncertain absolute scale and require calibration.
- Simulation results do not establish real-world performance or safety.
- The repository does not provide a complete autonomous-driving system, safety case, or operational monitoring.

## Safety statement

This is a research prototype and is **not for use in real vehicles or safety-critical decisions**. Do not use its outputs to control a vehicle or to make decisions affecting road users.
