# Experiment log

Record only completed runs with measured results in the table. Planned work is listed separately;
do not estimate or copy metrics from smoke tests.

| Run name | W&B link | Config | mAP50 | mAP50-95 |
|---|---|---|---:|---:|
| YOLO11n, 50 epochs, Kaggle Tesla T4 | Not recorded | imgsz 640, batch 16; 10,000 train / 2,000 val images | 0.434 | 0.234 |

Precision: **0.569**; recall: **0.409**.

| Class | AP50-95 |
|---|---:|
| Car | 0.415 |
| Truck | 0.369 |
| Bus | 0.341 |
| Motorcycle | 0.109 |
| Bicycle | 0.142 |
| Person | 0.208 |
| Rider | 0.117 |
| Traffic light | 0.163 |
| Traffic sign | 0.247 |

## Planned experiment (not run)

Train YOLO11n for 50 epochs at imgsz 640 and batch 16, keeping the other training settings
unchanged, but sample each training image containing motorcycle, rider, or bicycle objects three
times as often as under the current image sampling scheme. T4 runtime/cost: **to be measured**.
