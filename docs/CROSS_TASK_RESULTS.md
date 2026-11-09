# Cross-task BPTT / PPO results · 跨任务结果

Completed 2026-11-09: ten new XHand clips, both methods trained from scratch, seed 2026, 64M real transitions per run. Evaluation: 4096 first-frame episodes, seed 12345, 256 environments, fixed per-environment quotas.

| Task / 任务 | BPTT success | PPO success | BPTT train min | PPO train min |
|---|---:|---:|---:|---:|
| WineGlass / 酒杯 | 99.98% | 100.00% | 39.92 | 25.53 |
| WaterBottle / 水瓶 | 99.63% | 100.00% | 36.83 | 24.38 |
| Waffleiron / 华夫饼机 | 67.94% | 98.71% | 45.75 | 28.60 |
| CubeMedium / 方块交接 | 17.48% | 97.97% | 38.49 | 23.47 |
| Notebook / 笔记本 | 1.42% | 85.06% | 46.30 | 29.07 |
| Cuboid_01 | 0.37% | 0.78% | 31.44 | 16.88 |
| Cylinder_00 | 0.00% | 95.34% | 29.36 | 16.15 |
| Mixer / 搅拌器 | 0.00% | 56.93% | 43.44 | 27.89 |
| Ring_00 | 0.00% | 0.00% | 33.50 | 19.66 |
| CylinderMedium / 圆柱交接 | 0.00% | 0.00% | 39.05 | 23.85 |

BPTT used the unchanged Hammer e3 training core and configuration. No PPO policy initialization was used. Two new clips reached high success: WineGlass 4095/4096 and WaterBottle 4081/4096. The original Hammer result remains available in v0.1.0.

These are single-seed results at a 500-update budget, not the paper’s 5000-update benchmark or a multi-seed reliability claim. BPTT training process time was 1.51–1.86× PPO in this screen; this implementation has not established an efficiency advantage. Jobs used different nodes concurrently, so these times do not isolate BPTT kernel performance.

这轮 BPTT 沿用 Hammer e3，从头训练，未加载 PPO 权重。酒杯和水瓶有高成功率，但其他任务差距仍然明显。单 seed、500-update 筛选不能作为论文完整预算或多 seed 结论。此版本的实测耗时比 PPO 更长。

Machine-readable results and all 20 evaluation records: [results.json](../results/xhand_screen_20261109/results.json). Policies, manifests and evaluation receipts are included for [WineGlass](../results/wineglass_e3), [WaterBottle](../results/waterbottle_e3), [Waffleiron](../results/waffleiron_e3) and [CubeMedium](../results/cubemedium_e3). [Successful BPTT videos and recording commands](SUCCESSFUL_VIDEOS.md) cover these tasks plus Hammer.

## Data and environment / 数据与环境

Use the same pinned Isaac environment from [SETUP](SETUP.md). Clone current main for the new policies; v0.1.0 remains the frozen Hammer release.

```bash
git clone https://github.com/Hxxxz0/Supermimic-dex.git
cd Supermimic-dex
export CONTRACK_ISAACLAB_PATH=/absolute/path/to/IsaacLab
export CONTRACK_RESULTS_DIR="$PWD/outputs"
python3 scripts/supermimic/fetch_assets.py
bash scripts/supermimic/python.sh scripts/supermimic/prepare_assets.py --device cpu
python3 scripts/supermimic/verify_release.py
python3 scripts/supermimic/verify_cross_task_results.py
```

The downloader supplies all 26 retargeted references and robot resources. Direct references:

- [WineGlass HDF5](https://raw.githubusercontent.com/EmptyBlueBox/ConTrack/f5efe18e4b1f986566b252265c4bf63f3b67c6c6/data/xhand/grab-s1_wineglass_offhand_1-xarm7_xhand-156_516.h5)
- [WaterBottle HDF5](https://raw.githubusercontent.com/EmptyBlueBox/ConTrack/f5efe18e4b1f986566b252265c4bf63f3b67c6c6/data/xhand/grab-s1_waterbottle_offhand_1-xarm7_xhand-24_504.h5)

These are GRAB-derived clips and retain the same original data terms described in [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md).

## Evaluate and reproduce / 评估与复现

Evaluate the included WineGlass BPTT weights:

```bash
bash scripts/supermimic/python.sh scripts/supermimic/evaluate.py \
  --checkpoint results/wineglass_e3/policy.pt \
  --data data/xhand/grab-s1_wineglass_offhand_1-xarm7_xhand-156_516.h5 \
  --embodiment xhand --num-envs 256 --episodes 4096 --seed 12345 --device cuda:0 \
  --output "$CONTRACK_RESULTS_DIR/wineglass_evaluation.json"
```

For WaterBottle, replace the checkpoint with `results/waterbottle_e3/policy.pt`, the data with `data/xhand/grab-s1_waterbottle_offhand_1-xarm7_xhand-24_504.h5`, and choose a new output file.

Train either task from scratch using the same configuration:

```bash
bash scripts/supermimic/python.sh scripts/supermimic/train.py \
  --algorithm supermimic --preset e3 --config configs/hammer_best.json \
  --embodiment xhand --seed 2026 --budget-updates 500 --num-envs 8000 --device cuda:0 \
  --data data/xhand/grab-s1_wineglass_offhand_1-xarm7_xhand-156_516.h5
```

For the matched PPO control, use `--algorithm ppo` with the same data, environment count, seed and budget. The real task observations, actions and rewards remain unchanged. Imagined trajectories retain the original release’s differentiable contact and termination approximations.

## Sharpa follow-up / Sharpa 后续实验

All 13 RB-Y1A / Sharpa clips are scheduled for separate BPTT/PPO controls: 26 jobs, seed 2026, 6000 environments, 500 updates = 48M transitions per run, final evaluation 4096 episodes. This is the released Sharpa environment count, so it differs from the 64M XHand screen.

Sharpa experiment status is recorded in [experiment_status.json](../results/experiment_status.json). Ongoing jobs are not counted as completed results. The cluster runner is maintained on the `exp/sharpa-20261109` branch; its workspace paths and cloud resource configuration are deployment-specific.

Sharpa CubeMedium BPTT has a validated 99.7803% result (4087/4096), trained from scratch at 48M transitions. [Weights and evaluation](../results/sharpa_cubemedium_e3) and [video](SUCCESSFUL_VIDEOS.md) are included. The Sharpa campaign is ongoing; pending or failed startup cases have no final success rate.
