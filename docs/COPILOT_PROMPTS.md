# Copilot prompts for finishing the project

Open the repo in VS Code, Copilot Chat -> Agent mode. Paste one at a time.

1. **Orientation:** "Read README.md, docs/architecture.md and src/adpstack. Summarise the pipeline and list any code paths that were never executed on real data (they are marked `pragma: no cover`)."
2. **Data run:** "Help me run bdd100k_to_yolo and eda on my downloaded BDD100K data, fix any path issues, and fill the dataset section of docs/results.md from docs/figures/eda_summary.json. Never invent numbers."
3. **After Colab training:** "Here is detection_eval.json (paste). Update docs/results.md sections 3 and the failure analysis paragraph. Propose two evidence-based improvements."
4. **Lane analysis:** "Here are lane_compare.json and the qualitative grid. Write the 'when each approach fails' analysis in docs/results.md."
5. **Depth:** "Write scripts/eval_depth.py that loads data/carla/*/images + depth/*.npy, runs MiDaSDepth, fits scale/shift on half the frames, evaluates MAE/AbsRel/RMSE on the rest for MiDaS, pinhole and fused, and writes docs/figures/depth_eval.json."
6. **ROS2 debugging (on the VM):** "Here is the ros2 launch log (paste). Diagnose and fix the failing node without changing topic names."
7. **Polish:** "Run ruff and pytest, fix issues, and generate a clean conventional-commit history plan (feat:, test:, docs:) for the changes."
