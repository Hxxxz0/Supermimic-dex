#!/usr/bin/env python3
"""Evaluate PPO and SuperMimic exports with identical first-frame real-physics episodes."""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "source/ConTrack"))


def episode_quotas(episodes, num_envs, device):
    """Allocate trials before rollout so fast failures cannot dominate the sample."""
    import torch

    return (episodes + torch.arange(num_envs, device=device)) // num_envs


def joint_errors(q, reference, arm, finger):
    difference = q - reference
    return (
        difference[:, finger].square().mean(-1).sqrt(),
        difference[:, finger].abs().mean(-1),
        difference[:, arm].abs().mean(-1),
    )


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--embodiment", choices=("xhand", "sharpa"), default="xhand")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--episodes", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.checkpoint is None or args.data is None or args.output is None:
        parser.error("--checkpoint, --data and --output are required")
    if args.episodes <= 0 or args.num_envs < 2:
        parser.error("Invalid evaluation size")
    args.enable_cameras, args.headless = False, True
    os.environ["CONTRACK_XARM_XHAND_DATA"] = str(
        args.data.resolve()
        if args.embodiment == "xhand"
        else REPO / "data/xhand/dexterhand-Cylinder_00-fps_20-xarm7_xhand-3600_3700.h5"
    )
    os.environ["CONTRACK_RBY1A_SHARPA_DATA"] = str(
        args.data.resolve()
        if args.embodiment == "sharpa"
        else REPO / "data/sharpa/dexterhand-Cylinder_00-fps_20-rby1a_sharpa-3600_3700.h5"
    )
    from contrack_supermimic.experiment import prepare_output_links, sha256, git

    prepare_output_links(REPO)
    app = AppLauncher(args).app
    env = adapter = None
    try:
        import gymnasium as gym
        import torch
        import ConTrack.tasks  # noqa: F401
        from isaaclab_tasks.utils import load_cfg_from_registry
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl_contrack.modules import ActorCritic
        from contrack_supermimic.isaac_adapter import IsaacAdapter

        torch.manual_seed(args.seed)
        task_id = (
            "Isaac-Xarm-Xhand-Mimic-Manager-Play-v0"
            if args.embodiment == "xhand"
            else "Isaac-Rby1a-Sharpa-Mimic-Manager-Play-v0"
        )
        env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
        agent_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")
        env_cfg.scene.num_envs, env_cfg.seed = args.num_envs, args.seed
        env_cfg.sim.device = args.device or "cuda:0"
        # Play reset also resets per-env thresholds; lock both ends of the curriculum.
        env_cfg.object_pos_threshold_min = env_cfg.object_pos_threshold_max = 0.1
        env_cfg.object_rot_threshold_min = env_cfg.object_rot_threshold_max = 1.0
        env = gym.make(task_id, cfg=env_cfg, render_mode=None)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        adapter = IsaacAdapter(env.unwrapped)
        adapter.attach()
        obs = wrapped.get_observations()
        kwargs = agent_cfg.policy.to_dict()
        kwargs.pop("class_name", None)
        policy = ActorCritic(
            obs, {"policy": ["policy"], "critic": ["critic"]}, wrapped.num_actions, **kwargs
        ).to(wrapped.device)
        state = torch.load(args.checkpoint, map_location=wrapped.device, weights_only=False)
        policy.load_state_dict(state["model_state_dict"])
        policy.eval()
        device = wrapped.device
        quotas = episode_quotas(args.episodes, args.num_envs, device)
        completed = torch.zeros(args.num_envs, device=device, dtype=torch.long)
        total_completed = 0
        lengths = torch.zeros(args.num_envs, device=device)
        sums = torch.zeros(args.num_envs, 10, device=device)
        episode_results = []
        while total_completed < args.episodes:
            with torch.no_grad():
                adapter.begin()
                obs, _, _, _ = wrapped.step(policy.act_inference(obs))
                record = adapter.finish()
                task = adapter.task
                q, _, _ = task.codec.unpack(record["x1"])
                pos, rot = task.pose_error(record["x1"], record["memory"], stable=False)
                fingers, finger_mae, arm_mae = joint_errors(
                    q[:, : task.d.actions],
                    task.reference_q(record["memory"]),
                    task.arm,
                    task.finger,
                )
                ref_c = task.ref["contact"][task.index(record["memory"])]
                sim_c = record["contact"].bool()
                tp = (ref_c & sim_c).flatten(1).sum(-1)
                fp = (~ref_c & sim_c).flatten(1).sum(-1)
                fn = (ref_c & ~sim_c).flatten(1).sum(-1)
                joint = (ref_c & sim_c).float()
                point_sum = (
                    (
                        (record["points"] - task.ref["points"][task.index(record["memory"])]).norm(
                            dim=-1
                        )
                        * joint
                    )
                    .flatten(1)
                    .sum(-1)
                )
                sums += torch.stack(
                    (
                        pos.mean(-1),
                        rot.mean(-1),
                        fingers,
                        tp,
                        fp,
                        fn,
                        point_sum,
                        joint.flatten(1).sum(-1),
                        finger_mae,
                        arm_mae,
                    ),
                    -1,
                )
                lengths += 1
                ids = (record["done"] & (completed < quotas)).nonzero(as_tuple=False).flatten()
                if ids.numel():
                    success = (record["timeout"][ids] & ~record["terminated"][ids]).float()
                    progress = record["memory"][ids, 0] / (task.length - 1)
                    errors = sums[ids, :3] / lengths[ids, None]
                    counts = sums[ids, 3:8]
                    absolute_errors = sums[ids, 8:] / lengths[ids, None]
                    episode_results.append(
                        torch.cat(
                            (success[:, None], progress[:, None], errors, counts, absolute_errors),
                            -1,
                        ).cpu()
                    )
                    completed[ids] += 1
                    total_completed += ids.numel()
                    lengths[ids] = 0
                    sums[ids] = 0
        result = torch.cat(episode_results)[: args.episodes]
        counts = result[:, 5:10].sum(0)
        rate = result[:, 0].mean().item()
        z, n = 1.96, args.episodes
        center = (rate + z * z / (2 * n)) / (1 + z * z / n)
        half = z * ((rate * (1 - rate) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
        output = {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.checkpoint),
            "reference": str(args.data.resolve()),
            "reference_sha256": sha256(args.data),
            "evaluator_git_sha": git(REPO, "rev-parse", "HEAD"),
            "episode_allocation": "fixed_per_environment",
            "num_envs": args.num_envs,
            "seed": args.seed,
            "episodes": args.episodes,
            "success_rate": rate,
            "success_95ci": [max(0.0, center - half), min(1.0, center + half)],
            "progress": result[:, 1].mean().item(),
            "object_pos_error_m": result[:, 2].mean().item(),
            "object_rot_error_rad": result[:, 3].mean().item(),
            "finger_rmse_rad": result[:, 4].mean().item(),
            "finger_mae_rad": result[:, 10].mean().item(),
            "arm_mae_rad": result[:, 11].mean().item(),
            "metric_protocol": "ConTrack_sec4.3_episode_mean_pose_joint_and_pooled_contact",
            "contact_f1": float(
                2 * counts[0] / (2 * counts[0] + counts[1] + counts[2]).clamp_min(1)
            ),
            "contact_point_error_m": float(counts[3] / counts[4]) if counts[4] > 0 else None,
            "contact_point_samples": int(counts[4]),
            "thresholds": {"position_m": 0.1, "rotation_rad": 1.0},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n")
        print(json.dumps(output), flush=True)
    except BaseException:
        failure = traceback.format_exc()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".failure.json").write_text(
            json.dumps({"traceback": failure}) + "\n"
        )
        sys.stderr.write(failure)
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(1)
    finally:
        if adapter is not None:
            adapter.close()
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
