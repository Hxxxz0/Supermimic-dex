# Sharpa final results / Sharpa 最终结果

All 26 logical task/method cases have terminated; 21/26 have validated final evaluations. 38 cloud attempts include 12 diagnosed startup retries.
Each method/task: 6000 environments, 48M real transitions, training seed 2026; evaluation 4096 episodes, seed 12345.
XHand used 64M; this is not a matched-interaction cross-embodiment comparison.

| Task | BPTT success | PPO success | BPTT state | PPO state |
|---|---:|---:|---|---|
| arctic-s01_box_use_01-rby1a_sharpa-117_240 | not evaluated | not evaluated | failed | failed |
| arctic-s01_mixer_use_01-rby1a_sharpa-45_345 | 0.00% | 0.05% | evaluated | evaluated |
| arctic-s01_notebook_use_01-rby1a_sharpa-180_270 | 0.42% | 13.87% | evaluated | evaluated |
| arctic-s01_waffleiron_use_01-rby1a_sharpa-75_195 | 0.73% | 81.69% | evaluated | evaluated |
| dexterhand-Cuboid_00-fps_20-rby1a_sharpa-5880_5940 | 0.00% | 14.99% | evaluated | evaluated |
| dexterhand-Cuboid_01-fps_20-rby1a_sharpa-7824_7864 | 0.34% | not evaluated | evaluated | failed |
| dexterhand-Cylinder_00-fps_20-rby1a_sharpa-3600_3700 | 0.00% | 92.21% | evaluated | evaluated |
| dexterhand-Ring_00-fps_20-rby1a_sharpa-10240_10340 | 0.00% | 6.05% | evaluated | evaluated |
| grab-s1_cubemedium_offhand_1-rby1a_sharpa-180_540 | 99.78% | not evaluated | evaluated | failed |
| grab-s1_cylindermedium_offhand_1-rby1a_sharpa-268_628 | 0.00% | 0.00% | evaluated | evaluated |
| grab-s1_hammer_use_1-rby1a_sharpa-120_840 | 1.46% | 76.29% | evaluated | evaluated |
| grab-s1_waterbottle_offhand_1-rby1a_sharpa-24_504 | 0.76% | not evaluated | evaluated | failed |
| grab-s1_wineglass_offhand_1-rby1a_sharpa-156_516 | 0.17% | 55.74% | evaluated | evaluated |

No PPO initialization was used for BPTT. This is a single-seed screen, not a multi-seed reliability claim.
未完成评估的任务不计为成功，也不填入虚构成功率。训练预算、源码和权重哈希已核验。

[All evaluation records](../results/sharpa_screen_20261109/results.json)
