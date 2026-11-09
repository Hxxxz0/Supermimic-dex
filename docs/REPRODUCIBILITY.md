# Reproducibility and release scope

## Winning run

- Source: `a232096726ae6c9a6523aabe3b8cd7a156bd66a0`; the training manifest records an empty Git status.
- Algorithm: SuperMimic BPTT, trained from scratch; no PPO initialization or resumed checkpoint.
- Task: `grab-s1_hammer_use_1-xarm7_xhand-120_840`, two xArm7/XHand robots.
- Training: e3 / seed 2026 / 8000 environments / 64M real transitions.
- Evaluation: Play task, first-frame resets, seed 12345, 256 environments, preallocated per-environment quotas totaling 4096 episodes. Success is reference timeout without physical termination, using position 0.1 m and rotation 1 rad.

| Artifact | SHA256 |
| --- | --- |
| Included BPTT policy | `845db072f3598bd51e065ac933cd2d5f28c8a2b0f7425f24de7ab8d1ae0a57ba` |
| Hammer reference | `9771a4bd8cf813465af026d2f482bd0d9485b7a35313f1285c75d21a4756f183` |

The policy is the exact original export, including its original checkpoint schema. It can be loaded by the ConTrack policy interface. It is sufficient for inference/evaluation; it is not the full WM/replay/physics training-resume checkpoint.

Published manifests remove machine paths and GPU identifiers, preserving configuration, source hashes, original metrics and content hashes. Their original unsanitized JSON file hashes are recorded for provenance. [Training completion](../results/hammer_best/training_complete.json) includes real samples, Actor update count, timings and reward-parity measurements.

The launch JSON contains configuration overrides; `--preset e3` supplies the preset separately. The release verifier constructs the actual baseline `TrainConfig` from that JSON and checks its resolved value against the original manifest.

## What cleanup changes

The release starts from a fresh source snapshot, with no later experimental branches or historical worktrees. Original upstream task and RSL source files remain byte-identical. Core/utility Python formatting is normalized; two unused trainer imports are removed. Runtime results/Isaac paths are configurable, and PPO defaults to this repository's unchanged baseline implementation instead of a machine-specific sibling checkout. Two assigned lambda helpers are written as equivalent named functions. The calibration-selection CPU test receives synthetic temporary clip paths, so it can run without the 13 external reference files; its original metrics and assertions are preserved.

The BPTT algorithm, WM structure/loss, replay sampling, Critic/dual updates and winning hyperparameters are preserved. New additions are documentation, packaging, resource fetching, artifact/source verification, CI and policy-video recording. No failure head, CUDA Graph, small Actor, PPO initialization, or later WM-supervision experiments are included.

[Source contracts](frozen_source.json) record the original Git blobs and Python AST hashes. `verify_release.py` checks the executable AST against the frozen baseline, allowing only the explicitly enumerated runtime/helper/import transformations above; upstream task/RSL files must match byte-for-byte. It also checks the exact config and policy hash. This is an integrity check, not proof that new training has converged.

## Checks

```bash
ruff check .
ruff format --check .
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python3 scripts/supermimic/verify_release.py
python3 scripts/supermimic/fetch_assets.py --verify-only
```

The original frozen CPU suite has 25 tests. Native Isaac loading/rendering and replay evaluation are separate from CPU checks. Training from this cleaned release has not been rerun to 64M samples; the reported result remains the original measured run. Native solver state, hardware and conversion changes prevent a promise of bitwise identical retraining.

Optional `suite.py`, `evaluate_curves.py`, `evaluate_wm.py`, `monitor.py`, `watch_run.py` and `report.py` are retained as compact baseline utilities. The multi-task suite requires additional reference clips; it does not imply multi-task results for this release. Generated logs, videos, simulator resources and checkpoints from new experiments must stay outside tracked source.
