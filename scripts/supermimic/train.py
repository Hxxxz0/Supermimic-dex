#!/usr/bin/env python3
"""Unified launcher: original ConTrack PPO or donor-style differentiable training."""

import argparse
import json
import os
import random
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "source" / "ConTrack"))

from contrack_supermimic.config import TrainConfig
from contrack_supermimic.experiment import RESULT_ROOT, manifest, prepare_output_links


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("ppo", "supermimic"), default="supermimic")
    parser.add_argument("--embodiment", choices=("xhand", "sharpa"), default="xhand")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget-updates", type=int, default=5000)
    parser.add_argument("--preset", choices=("e1", "e2", "e3", "reference"), default="e1")
    parser.add_argument("--config", type=Path, help="JSON overrides for SuperMimic config")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument(
        "--source-repo",
        type=Path,
        help="Original task/PPO checkout, separate from the training harness",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.data is None:
        parser.error("--data is required")
    source_repo = args.source_repo or REPO
    if not source_repo.is_dir():
        parser.error(f"Missing source checkout {source_repo}")
    sys.path.insert(0, str(source_repo))
    sys.path.insert(0, str(source_repo / "source/ConTrack"))
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Use one GPU per process")
    args.data = args.data.resolve()
    if not args.data.is_file():
        parser.error(f"Missing reference {args.data}")
    num_envs = args.num_envs or (
        32 if args.smoke else (8000 if args.embodiment == "xhand" else 6000)
    )
    if num_envs < 2:
        parser.error("Need >=2 environments")
    args.enable_cameras = False
    args.headless = True
    os.environ[
        "CONTRACK_XARM_XHAND_DATA" if args.embodiment == "xhand" else "CONTRACK_RBY1A_SHARPA_DATA"
    ] = str(args.data)
    # Package registration imports both embodiments; upstream XHand's default path
    # omits data/xhand. Supply a valid default for the other registered embodiment.
    os.environ.setdefault(
        "CONTRACK_XARM_XHAND_DATA",
        str(REPO / "data/xhand/dexterhand-Cylinder_00-fps_20-xarm7_xhand-3600_3700.h5"),
    )
    os.environ.setdefault(
        "CONTRACK_RBY1A_SHARPA_DATA",
        str(REPO / "data/sharpa/dexterhand-Cylinder_00-fps_20-rby1a_sharpa-3600_3700.h5"),
    )
    prepare_output_links(source_repo)
    overrides = {
        "seed": args.seed,
        "budget_updates": args.budget_updates,
        "compile": not args.no_compile,
    }
    if args.smoke:
        overrides.update(
            budget_updates=8,
            warmup_steps=16,
            wm_only_updates=2,
            warmup_wm_batches=1,
            wm_hidden=(64, 64),
            wm_batch=32,
            wm_microbatch=32,
            actor_batch=32,
            actor_microbatch=32,
            horizon=8,
            replay_bytes=256 * 1024**2,
            compile=False,
            log_interval=1,
            critic_epochs=1,
        )
    if args.config:
        overrides.update(json.loads(args.config.read_text()))
    if "wm_hidden" in overrides:
        overrides["wm_hidden"] = tuple(overrides["wm_hidden"])
    cfg = TrainConfig.from_preset(args.preset, **overrides)
    if not args.run_dir:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.run_dir = (
            RESULT_ROOT
            / "runs"
            / f"{args.data.stem}/{args.algorithm}-{args.preset}-seed{args.seed}-{stamp}"
        )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if (args.run_dir / "manifest.json").exists() and not args.resume:
        parser.error("Run directory already has a manifest; choose a new directory or --resume")
    app = AppLauncher(args).app
    env = None
    try:
        import gymnasium as gym
        import numpy as np
        import torch
        import ConTrack.tasks  # noqa: F401
        from isaaclab_tasks.utils import load_cfg_from_registry
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab.utils.io import dump_yaml
        from rsl_rl_contrack.env.cons_mimic_reward_wrapper import ConsMimicRewardWrapper
        from contrack_supermimic.isaac_adapter import IsaacAdapter

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        task_id = (
            "Isaac-Xarm-Xhand-Mimic-Manager-v0"
            if args.embodiment == "xhand"
            else "Isaac-Rby1a-Sharpa-Mimic-Manager-v0"
        )
        env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
        agent_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")
        env_cfg.scene.num_envs = num_envs
        env_cfg.seed = agent_cfg.seed = args.seed
        env_cfg.sim.device = agent_cfg.device = args.device or "cuda:0"
        agent_cfg.max_iterations = cfg.budget_updates
        agent_cfg.logger = "tensorboard"
        resolved = agent_cfg.to_dict() if args.algorithm == "ppo" else cfg.to_dict()
        if args.algorithm == "ppo":
            resolved["infrastructure"] = {"compile": cfg.compile, "tf32": True, "cameras": False}
        manifest(
            REPO,
            args.data,
            args.algorithm,
            resolved,
            num_envs,
            args.seed,
            args.run_dir,
            args.resume,
            args.smoke,
            source_repo,
        )
        dump_yaml(str(args.run_dir / "params/env.yaml"), env_cfg)
        dump_yaml(str(args.run_dir / "params/agent.yaml"), agent_cfg)
        (args.run_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2) + "\n")
        env = gym.make(task_id, cfg=env_cfg, render_mode=None)
        isaac = env.unwrapped
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        wrapped = ConsMimicRewardWrapper(wrapped, isaac)
        adapter = IsaacAdapter(isaac, cfg.pos_soft_width, cfg.rot_soft_width)
        (args.run_dir / "descriptor.json").write_text(
            json.dumps(adapter.d.to_dict(), indent=2) + "\n"
        )
        if args.algorithm == "ppo":
            from rsl_rl_contrack.runners import OnPolicyRunner
            from contrack_supermimic.experiment import PPOAudit

            audit = PPOAudit(wrapped, adapter, args.run_dir, cfg.log_interval * 16)
            runner = OnPolicyRunner(audit, agent_cfg.to_dict(), str(args.run_dir), agent_cfg.device)
            if cfg.compile and str(agent_cfg.device).startswith("cuda"):
                # Compile forward methods, preserving module/state_dict names and
                # the original PPO algorithm, Gaussian std and checkpoint interface.
                runner.alg.policy.actor.forward = torch.compile(runner.alg.policy.actor.forward)
                runner.alg.policy.critic.forward = torch.compile(runner.alg.policy.critic.forward)
            if args.resume:
                runner.load(str(args.resume))
            runner.learn(
                num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
            )
            audit.log()
            adapter.close()
        else:
            from rsl_rl_contrack.modules import ActorCritic
            from contrack_supermimic.trainer import SuperMimicTrainer

            kwargs = agent_cfg.policy.to_dict()
            kwargs.pop("class_name", None)
            policy = ActorCritic(
                wrapped.get_observations(),
                {"policy": ["policy"], "critic": ["critic"]},
                wrapped.num_actions,
                **kwargs,
            ).to(agent_cfg.device)
            trainer = SuperMimicTrainer(wrapped, adapter, policy, cfg, args.run_dir)
            if args.resume:
                trainer.load(args.resume)
            trainer.learn()
    except BaseException:
        failure = traceback.format_exc()
        (args.run_dir / "failure.json").write_text(json.dumps({"traceback": failure}) + "\n")
        sys.stderr.write(failure)
        sys.stderr.flush()
        sys.stdout.flush()
        # Kit's native shutdown may terminate Python with status zero before an
        # exception unwinds. Preserve the failure marker and a nonzero exit code.
        os._exit(1)
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
