# BPTT success videos / BPTT 成功视频

These videos use the evaluated, from-scratch SuperMimic BPTT policies in native Isaac physics. The table reports the separate 4096-episode evaluation, training seed 2026 and evaluation seed 12345. The first five rows use xArm7/XHand; the last uses RB-Y1A/Sharpa.

| Task / 任务 | Video / 视频 | Full evaluation success | Policy / 权重 |
|---|---|---:|---|
| Hammer / 锤子 | [3 consecutive successes, 18 seconds](media/hammer_bptt_best.mp4) | 99.9756% | [policy.pt](../results/hammer_best/policy.pt) |
| WineGlass / 酒杯 | [3 successful episodes](media/wineglass_bptt_success.mp4) | 99.9756% | [policy.pt](../results/wineglass_e3/policy.pt) |
| WaterBottle / 水瓶 | [3 successful episodes](media/waterbottle_bptt_success.mp4) | 99.6338% | [policy.pt](../results/waterbottle_e3/policy.pt) |
| Waffleiron / 华夫饼机 | [3 successful episodes](media/waffleiron_bptt_success.mp4) | 67.9443% | [policy.pt](../results/waffleiron_e3/policy.pt) |
| CubeMedium / 方块交接 | [3 successful episodes](media/cubemedium_bptt_success.mp4) | 17.4805% | [policy.pt](../results/cubemedium_e3/policy.pt) |
| CubeMedium / Sharpa 方块交接 | [3 successful episodes](media/sharpa_cubemedium_bptt_success.mp4) | 99.7803% | [policy.pt](../results/sharpa_cubemedium_e3/policy.pt) |

The five new videos retain successful examples. Their receipts include every recorded attempt, rejected attempts, any environment prescan, camera, seed, renderer revision, and weight/reference/video hashes. Waffleiron and CubeMedium select a successful environment from a bounded 64-environment prescan; CubeMedium requires three consecutive prescan successes. These selections do not estimate success rate; use the full evaluation column above.

新视频展示筛选后的成功回合，每段 3 回合。华夫饼机与方块交接会先在 64 个环境中寻找成功环境，再录制该环境的完整回合。表中的成功率来自独立 4096 回合评估；方块交接的整体成功率仍然只有 17.48%。

XHand training used 64M transitions per task; Sharpa used 48M. These are separate single-seed screens, with different embodiments and interaction budgets.

Recording receipts: [WineGlass](media/wineglass_bptt_success.json), [WaterBottle](media/waterbottle_bptt_success.json), [Waffleiron](media/waffleiron_bptt_success.json), [CubeMedium](media/cubemedium_bptt_success.json), [Sharpa CubeMedium](media/sharpa_cubemedium_bptt_success.json).

## Reproduce / 复现

Prepare the pinned Isaac environment, data and robot assets using [SETUP](SETUP.md). The downloader includes all these retargeted ConTrack clips. Original source links:

- [WineGlass data](https://raw.githubusercontent.com/EmptyBlueBox/ConTrack/f5efe18e4b1f986566b252265c4bf63f3b67c6c6/data/xhand/grab-s1_wineglass_offhand_1-xarm7_xhand-156_516.h5)
- [WaterBottle data](https://raw.githubusercontent.com/EmptyBlueBox/ConTrack/f5efe18e4b1f986566b252265c4bf63f3b67c6c6/data/xhand/grab-s1_waterbottle_offhand_1-xarm7_xhand-24_504.h5)
- [Waffleiron data](https://raw.githubusercontent.com/EmptyBlueBox/ConTrack/f5efe18e4b1f986566b252265c4bf63f3b67c6c6/data/xhand/arctic-s01_waffleiron_use_01-xarm7_xhand-75_195.h5)
- [CubeMedium data](https://raw.githubusercontent.com/EmptyBlueBox/ConTrack/f5efe18e4b1f986566b252265c4bf63f3b67c6c6/data/xhand/grab-s1_cubemedium_offhand_1-xarm7_xhand-180_540.h5)

```bash
bash scripts/supermimic/python.sh scripts/supermimic/record_video.py \
  --checkpoint results/wineglass_e3/policy.pt \
  --data data/xhand/grab-s1_wineglass_offhand_1-xarm7_xhand-156_516.h5 \
  --output "$CONTRACK_RESULTS_DIR/wineglass_bptt_success.mp4" \
  --title WineGlass --episodes 3 --seed 12345 --device cuda:0 \
  --successes-only --max-attempts 60
```

For Sharpa CubeMedium, use `results/sharpa_cubemedium_e3/policy.pt`, `data/sharpa/grab-s1_cubemedium_offhand_1-rby1a_sharpa-180_540.h5`, and `--embodiment sharpa`.

For Waffleiron/CubeMedium, replace the checkpoint, data, output and title, and add `--num-envs 64 --select-success-env`. CubeMedium also uses `--prescan-successes 3`. For consecutive episodes without filtering or a prescan, omit `--successes-only` and `--select-success-env`.

```bash
python3 scripts/supermimic/verify_recorded_videos.py
```

The real task action, observation and reward definitions and the trained weights are preserved. A successful episode reaches the reference timeout without failure termination, with position/rotation thresholds fixed at 0.1 m and 1 rad. This pose-based criterion does not require perfect contact tracking. [All task results](CROSS_TASK_RESULTS.md).
