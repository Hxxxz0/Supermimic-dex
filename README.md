# Supermimic-dex

SuperMimic-style backpropagation through time (BPTT) for the released [ConTrack](https://github.com/EmptyBlueBox/ConTrack) dexterous motion-tracking tasks, using real Isaac physics for data collection and evaluation.

This release packages the **Hammer BPTT policy with 99.9756% recorded success (4095 / 4096 episodes)**. Its training source is `a232096726ae6c9a6523aabe3b8cd7a156bd66a0`, preset `e3`, seed `2026`. It uses **two xArm7/XHand robots**. The policy weights are included in [results/hammer_best](results/hammer_best).

[中文说明](README_zh.md) · [Setup](docs/SETUP.md) · [BPTT / WM design](docs/METHOD.md) · [Reproducibility](docs/REPRODUCIBILITY.md)

## Recorded Hammer result

| Item | Value |
| --- | --- |
| Algorithm | SuperMimic-style BPTT, trained from scratch |
| Success | 4095 / 4096 = 99.9756% |
| 95% success interval | 99.8618%–99.9957% |
| Mean progress | 99.9954% |
| Mean object position error | 0.02363 m |
| Mean object rotation error | 0.21057 rad |
| Contact F1 | 0.69264 |
| Success thresholds | Object position 0.1 m; rotation 1 rad |
| Training budget | 64 million real transitions, 8000 environments |
| Training process time | 2619.86 s on one RTX 5090 |

These are measurements of one selected Hammer calibration run, not an aggregate over tasks or training seeds. The pose-based success criterion does not require perfect contact tracking. This result does not establish faster learning than PPO at matched quality. [Evaluation receipt](results/hammer_best/evaluation.json).

## Quick start

Install the pinned Isaac environment first; see [setup](docs/SETUP.md). With an existing Isaac image:

```bash
git clone https://github.com/Hxxxz0/Supermimic-dex.git
cd Supermimic-dex
export CONTRACK_ISAACLAB_PATH=/absolute/path/to/IsaacLab
export CONTRACK_RESULTS_DIR=/absolute/path/to/experiment-results
bash scripts/supermimic/python.sh -m pip install --no-deps -e .
python3 scripts/supermimic/fetch_assets.py
bash scripts/supermimic/python.sh scripts/supermimic/prepare_assets.py --device cpu
python3 scripts/supermimic/verify_release.py
```

The resource downloader pins and checks every file against upstream. Review the [resource terms](THIRD_PARTY_NOTICES.md), including the GRAB research-use conditions, before downloading/using the Hammer reference.

Reproduce the winning training configuration:

```bash
bash scripts/supermimic/python.sh scripts/supermimic/train.py \
  --algorithm supermimic --preset e3 --seed 2026 --budget-updates 500 \
  --config configs/hammer_best.json --num-envs 8000 --device cuda:0 \
  --data data/xhand/grab-s1_hammer_use_1-xarm7_xhand-120_840.h5
```

Evaluate the included BPTT policy with the original protocol:

```bash
bash scripts/supermimic/python.sh scripts/supermimic/evaluate.py \
  --checkpoint results/hammer_best/policy.pt \
  --data data/xhand/grab-s1_hammer_use_1-xarm7_xhand-120_840.h5 \
  --episodes 4096 --num-envs 256 --seed 12345 --device cuda:0 \
  --output "$CONTRACK_RESULTS_DIR/hammer-evaluation.json"
```

The exact training command is an expected reproduction recipe, not a promise of bitwise identical retraining. PhysX and GPU numerical differences can affect trajectories.

## Recording a policy video

[![Hammer BPTT policy rollout](docs/media/hammer_preview.png)](https://github.com/Hxxxz0/Supermimic-dex/raw/refs/heads/main/docs/media/hammer_bptt_best.mp4)

[Watch/download the 18-second BPTT video](docs/media/hammer_bptt_best.mp4) · [Recording receipt](docs/media/hammer_bptt_best.json). Three consecutive episodes succeeded; this is not a replacement for the full evaluation.

```bash
bash scripts/supermimic/python.sh scripts/supermimic/record_video.py \
  --checkpoint results/hammer_best/policy.pt \
  --data data/xhand/grab-s1_hammer_use_1-xarm7_xhand-120_840.h5 \
  --episodes 3 --device cuda:0 --output "$CONTRACK_RESULTS_DIR/hammer-bptt.mp4"
```

This records consecutive episodes from the trained BPTT policy, with no success filtering, in real Isaac physics. The video receipt records hashes, seed, camera and episode outcomes. It is illustrative; the success rate above comes from the separate 4096-episode evaluation.

## Development

```bash
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python scripts/supermimic/verify_release.py
```

The CPU suite checks BPTT finite differences, frozen parameters, microbatch loss scaling, recursive WM fitting, replay episode boundaries, task reward parity, timeout bootstrapping and fixed evaluation quotas. CI repeats these checks. Isaac integration and rendering require the native simulator and a compatible GPU environment.

Code is BSD-3-Clause. Upstream code attribution and separate asset/data terms are retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [licenses](licenses).
