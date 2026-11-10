# 10-minute presentation outline

| Min | Slide | Content |
|---|---|---|
| 0:00-1:00 | Problem | Perception is the first link of the autonomy chain; goal: detect, track, localize in lane, measure distance, and act - with measurable latency. |
| 1:00-2:30 | Design | Architecture diagram (docs/architecture.md); dev laptop vs Colab vs GPU VM split; why ROS2 + CARLA. |
| 2:30-4:00 | Data & detection | BDD100K 10K + CARLA synthetic; class imbalance and mitigations; mAP table; night/rain failure slices. |
| 4:00-5:00 | Tracking & lanes | ByteTrack MOTA/IDF1; classical vs U-Net lane IoU/FPS with the qualitative grid. |
| 5:00-6:00 | Depth & risk | MiDaS calibration, pinhole fusion, AbsRel; TTC -> risk -> AEB logic. |
| 6:00-7:30 | Optimization | ONNX -> TensorRT FP16/INT8 before/after latency chart; INT8 accuracy drop. |
| 7:30-9:00 | Live/recorded demo | CARLA pedestrian crossing: HUD with distance/TTC, braking; scenario metrics table. |
| 9:00-10:00 | Challenges & next steps | Challenges: domain gap, depth scale, latency; Next: multi-camera/LiDAR fusion, learned planner, Jetson deployment, DeepSORT/ReID comparison. |

Rehearse to 9:30. Keep one slide per number table. Every claim must point to a file in the repo or a W&B run.
