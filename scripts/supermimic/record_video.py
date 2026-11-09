#!/usr/bin/env python3
"""Record native policy episodes, optionally retaining only verified successes."""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "source/ConTrack"))


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--title", help="Task label displayed on the video")
    parser.add_argument("--embodiment", choices=("xhand", "sharpa"), default="xhand")
    parser.add_argument("--successes-only", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=30)
    parser.add_argument("--eye", type=float, nargs=3, default=(0.9, -0.85, 0.7))
    parser.add_argument("--lookat", type=float, nargs=3, default=(0.4, 0.0, 0.15))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.episodes < 1 or args.fps < 1:
        parser.error("--episodes and --fps must be positive")
    if args.successes_only and args.max_attempts < args.episodes:
        parser.error("--max-attempts must be at least --episodes")
    args.headless = args.enable_cameras = True
    os.environ["CONTRACK_XARM_XHAND_DATA"] = str(
        args.data.resolve() if args.embodiment == "xhand" else
        REPO / "data/xhand/dexterhand-Cylinder_00-fps_20-xarm7_xhand-3600_3700.h5"
    )
    os.environ["CONTRACK_RBY1A_SHARPA_DATA"] = str(
        args.data.resolve() if args.embodiment == "sharpa" else
        REPO / "data/sharpa/dexterhand-Cylinder_00-fps_20-rby1a_sharpa-3600_3700.h5"
    )
    from contrack_supermimic.experiment import prepare_output_links, sha256, git

    prepare_output_links(REPO)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    app = AppLauncher(args).app
    env = adapter = writer = None
    try:
        import gymnasium as gym
        import imageio.v2 as imageio
        import numpy as np
        import torch
        from PIL import Image, ImageDraw, ImageFont

        import ConTrack.tasks  # noqa: F401
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils import load_cfg_from_registry

        from contrack_supermimic.isaac_adapter import IsaacAdapter
        from rsl_rl_contrack.modules import ActorCritic

        torch.manual_seed(args.seed)
        task_id = ("Isaac-Xarm-Xhand-Mimic-Manager-Play-v0" if args.embodiment == "xhand"
                   else "Isaac-Rby1a-Sharpa-Mimic-Manager-Play-v0")
        env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
        agent_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")
        env_cfg.scene.num_envs, env_cfg.seed = 2, args.seed
        env_cfg.sim.device = args.device or "cuda:0"
        env_cfg.object_pos_threshold_min = env_cfg.object_pos_threshold_max = 0.1
        env_cfg.object_rot_threshold_min = env_cfg.object_rot_threshold_max = 1.0
        env_cfg.viewer.resolution = (1280, 720)
        env_cfg.viewer.eye, env_cfg.viewer.lookat = tuple(args.eye), tuple(args.lookat)
        env_cfg.viewer.origin_type, env_cfg.viewer.env_index = "env", 0
        env = gym.make(task_id, cfg=env_cfg, render_mode="rgb_array")
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
        algorithm = (state.get("infos") or {}).get("algorithm", "ppo")
        method = "SuperMimic BPTT" if algorithm == "supermimic" else "PPO"
        title = args.title or args.data.stem
        policy.load_state_dict(state["model_state_dict"])
        policy.eval()
        stride = max(1, round(1 / (env.unwrapped.step_dt * args.fps)))
        actual_fps = 1 / (env.unwrapped.step_dt * stride)
        writer = imageio.get_writer(
            args.output,
            format="FFMPEG",
            fps=actual_fps,
            codec="libx264",
            quality=8,
            macro_block_size=2,
            ffmpeg_params=["-movflags", "+faststart"],
        )
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 22)
        except OSError:
            font = ImageFont.load_default()
        episodes, attempts, steps, frames, episode_steps = [], [], 0, 0, 0
        episode_frames = []
        attempt_limit = args.max_attempts if args.successes_only else args.episodes
        record_stride_steps = 0

        def append_frame(frame):
            nonlocal frames
            writer.append_data(frame)
            if frames in (0, 45, 90):
                Image.fromarray(frame).save(
                    args.output.with_name(f"{args.output.stem}_frame_{frames:04d}.png")
                )
            frames += 1
        # Bounded even if the simulator's termination logic fails unexpectedly.
        max_steps = attempt_limit * (int(adapter.task.length / adapter.task.frame_step) + 10)
        while (len(episodes) < args.episodes and len(attempts) < attempt_limit
               and steps < max_steps):
            with torch.inference_mode():
                adapter.begin()
                obs, _, _, _ = wrapped.step(policy.act_inference(obs))
                record = adapter.finish()
            if not bool(record["valid"][0]):
                raise RuntimeError("Non-finite native state in the recorded environment")
            if record_stride_steps % stride == 0:
                frame = env.render()
                if frame is None:
                    raise RuntimeError("Isaac returned no RGB frame; check RTX/Vulkan rendering")
                picture = Image.fromarray(frame)
                draw = ImageDraw.Draw(picture)
                draw.rectangle((0, 0, 1280, 70), fill=(16, 20, 28))
                draw.text(
                    (18, 9),
                    f"{title} | {method} | trained policy / real Isaac physics",
                    font=font,
                    fill="white",
                )
                draw.text(
                    (18, 38),
                    (f"Attempt {len(attempts) + 1} | selected success {len(episodes) + 1}/{args.episodes}"
                     if args.successes_only else f"Consecutive episode {len(episodes) + 1}/{args.episodes}")
                    + f"  |  seed {args.seed}  |  {episode_steps * env.unwrapped.step_dt:.2f} s",
                    font=font,
                    fill=(190, 210, 230),
                )
                if args.successes_only:
                    episode_frames.append(np.asarray(picture).copy())
                else:
                    append_frame(np.asarray(picture))
            steps += 1
            record_stride_steps += 1
            episode_steps += 1
            if bool(record["done"][0]):
                outcome = dict(attempt=len(attempts) + 1, steps=episode_steps,
                               success=bool(record["timeout"][0] and not record["terminated"][0]))
                attempts.append(outcome)
                if not args.successes_only or outcome["success"]:
                    episodes.append(outcome | dict(episode=len(episodes) + 1))
                    if args.successes_only:
                        for frame in episode_frames:
                            append_frame(frame)
                print(json.dumps(outcome), flush=True)
                episode_frames.clear()
                episode_steps = 0
                record_stride_steps = 0
        writer.close()
        writer = None
        if len(episodes) != args.episodes:
            args.output.with_suffix(".attempts.json").write_text(json.dumps(attempts, indent=2) + "\n")
            raise RuntimeError("Recording did not complete the requested episode count within the attempt limit")
        receipt = {
            "task": title,
            "embodiment": args.embodiment,
            "algorithm": algorithm,
            "renderer_git_sha": git(REPO, "rev-parse", "HEAD"),
            "checkpoint_sha256": sha256(args.checkpoint),
            "reference_sha256": sha256(args.data),
            "evaluation_seed": args.seed,
            "environment_index": 0,
            "num_envs": 2,
            "selection": ("first successful episodes; all attempts recorded" if args.successes_only
                          else "first consecutive episodes; no success filtering"),
            "episodes": episodes,
            "attempts": attempts,
            "thresholds": {"position_m": 0.1, "rotation_rad": 1.0},
            "fps": actual_fps,
            "frames": frames,
            "duration_seconds": frames / actual_fps,
            "video_sha256": sha256(args.output),
            "camera_eye": args.eye,
            "camera_lookat": args.lookat,
            "note": "Illustrative policy rollout; the reported success rate comes from a separate 4096-episode evaluation.",
        }
        args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt), flush=True)
    except BaseException:
        args.output.with_suffix(".failure.txt").write_text(traceback.format_exc())
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    finally:
        if writer is not None:
            writer.close()
        if adapter is not None:
            adapter.close()
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
