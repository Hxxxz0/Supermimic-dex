#!/usr/bin/env python3
"""Held-out open-loop world-model errors at 1/8/16/32 real control steps."""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main():
    import torch
    from contrack_supermimic.config import TrainConfig
    from contrack_supermimic.model import WorldModel
    from contrack_supermimic.replay import ReplayBuffer
    from contrack_supermimic.state import TaskDescriptor, quaternion_error

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    if args.batch <= 0:
        parser.error("--batch must be positive")
    torch.manual_seed(args.seed)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    d = TaskDescriptor(**state["descriptor"])
    cfg = TrainConfig(**state["config"])
    wm = WorldModel(d, cfg.wm_hidden, cfg.amp).to(args.device)
    wm.load_state_dict(state["wm_state_dict"])
    wm.eval()
    saved = state["replay"]
    example = {k: v[0].to(args.device) for k, v in saved["data"].items()}
    context = saved["context"].to(args.device)
    replay = ReplayBuffer(
        example,
        context.shape[0],
        cfg.replay_bytes,
        cfg.replay_max_steps,
        cfg.holdout_fraction,
        cfg.wm_window * 2,
        static_context=context,
    )
    replay.load_state_dict(saved)
    length = min(32, replay.size)
    batch = replay.sample(args.batch, length, holdout=True)
    x = batch["x"][:, 0]
    result = {}
    with torch.no_grad():
        for t in range(length):
            x, logits, points = wm(x, batch["u"][:, t], batch["context"][:, t])
            if t + 1 not in (1, 8, 16, 32):
                continue
            mask = batch["mask"][:, t]
            if not mask.any():
                result[str(t + 1)] = {"valid_samples": 0}
                continue
            q, v, obj = wm.codec.unpack(x[mask])
            tq, tv, target = wm.codec.unpack(batch["x1"][mask, t])
            pred, truth = logits[mask] >= 0, batch["contact"][mask, t].bool()
            tp, fp, fn = (pred & truth).sum(), (pred & ~truth).sum(), (~pred & truth).sum()
            point_error = (points[mask] - batch["points"][mask, t]).norm(dim=-1)
            result[str(t + 1)] = {
                "valid_samples": int(mask.sum()),
                "joint_position_rmse_rad": float((q - tq).square().mean().sqrt()),
                "joint_velocity_rmse_rad_s": float((v - tv).square().mean().sqrt()),
                "object_position_rmse_m": float(
                    (obj[..., :3] - target[..., :3]).square().mean().sqrt()
                ),
                "object_rotation_error_rad": float(
                    quaternion_error(obj[..., 3:7], target[..., 3:7]).mean()
                ),
                "contact_f1": float(2 * tp / (2 * tp + fp + fn).clamp_min(1)),
                "contact_point_error_m": float(point_error[truth].mean()) if truth.any() else None,
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
