#!/usr/bin/env bash
# Run all CARLA scenarios sequentially on the GPU VM (CARLA server must already be running).
# usage: scripts/run_scenarios.sh models/best.onnx
set -euo pipefail
WEIGHTS=${1:-models/best.onnx}
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
for S in pedestrian_crossing lead_braking cut_in night_rain; do
  echo "=== scenario: $S ==="
  ros2 launch adp_bringup full_stack.launch.py scenario:=$S weights:=$WEIGHTS duration_s:=30.0 &
  PID=$!
  sleep 40; kill -INT $PID || true; wait $PID || true
  sleep 3
done
echo "metrics in outputs/scenarios/*.json"
