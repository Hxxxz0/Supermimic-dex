import os
import hashlib
import random
import time
from pathlib import Path

import numpy as np
import torch

from .replay import ReplayBuffer


ENV_TENSORS = (
    "frame_idx",
    "frame_idx_f",
    "episode_length_buf",
    "prev_arm_qvel",
    "prev_finger_qvel",
    "prev_object_lin_vel",
    "prev_object_ang_vel",
    "object_pos_threshold",
    "object_rot_threshold",
    "reset_lib_state",
    "reset_lib_track_len",
    "reset_lib_track_len_best",
    "reset_lib_terminate_index_count",
    "reset_lib_reset_index_count",
    "episode_start_frame",
    "episode_valid",
    "episode_filled",
    "reset_buf",
    "reset_terminated",
    "reset_time_outs",
)


def reference_hash(adapter):
    path = Path(adapter.env.cfg.reference_path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def environment_state(adapter):
    env = adapter.env
    tensors = {
        name: getattr(env, name).detach().cpu() for name in ENV_TENSORS if hasattr(env, name)
    }
    if hasattr(env, "episode_states"):
        tensors["episode_states"] = torch.where(
            env.episode_filled[..., None], env.episode_states, 0.0
        ).cpu()
    return {
        "physical": adapter.state().detach().cpu(),
        "tensors": tensors,
        "scalars": {n: getattr(env, n) for n in ("common_step_counter", "_sim_step_counter")},
        "adapter_episode": adapter.episode.cpu(),
        "context": adapter.context.cpu(),
        "sensor_contacts": {
            key: env.scene[key].data.contact_pos_w.detach().cpu()
            for keys in adapter.contact_keys
            for key in keys
        },
    }


def restore_environment(adapter, state):
    env = adapter.env
    x = state["physical"].to(adapter.device)
    q, v, obj = adapter.task.codec.unpack(x)
    offset = 0
    for name, ids in adapter.read_groups:
        n = len(ids)
        env.scene[name].write_joint_state_to_sim(
            q[:, offset : offset + n], v[:, offset : offset + n], joint_ids=ids
        )
        offset += n
    obj = obj.clone()
    obj[..., :3] += env.scene.env_origins[:, None, :]
    env.scene["objects"].write_object_state_to_sim(obj)
    for name, value in state["tensors"].items():
        if hasattr(env, name):
            getattr(env, name).copy_(value)
        else:
            setattr(env, name, value.to(adapter.device).clone())
    for name, value in state["scalars"].items():
        setattr(env, name, value)
    adapter.episode.copy_(state["adapter_episode"])
    adapter.context.copy_(state["context"])
    env.scene.update(dt=0.0)
    for key, value in state.get("sensor_contacts", {}).items():
        env.scene[key].data.contact_pos_w.copy_(value)


def save_checkpoint(trainer, path):
    path = Path(path)
    state = {
        "format_version": 1,
        "descriptor": trainer.d.to_dict(),
        "config": trainer.cfg.to_dict(),
        "reference_sha256": reference_hash(trainer.adapter),
        "model_state_dict": trainer.policy.state_dict(),
        "wm_state_dict": trainer.wm.state_dict(),
        "optimizers": {k: v.state_dict() for k, v in trainer.optimizers.items()},
        "dual": trainer.dual.state_dict(),
        "replay": trainer.replay.state_dict(),
        "environment": environment_state(trainer.adapter),
        "observations": {k: v.detach().cpu() for k, v in trainer.observations.items()},
        "progress": {
            k: getattr(trainer, k)
            for k in (
                "vector_steps",
                "iteration",
                "policy_updates",
                "collect_accumulator",
                "last_log",
                "last_save",
                "parity_count",
                "parity_max",
            )
        },
        "elapsed": trainer.elapsed + time.perf_counter() - trainer.started,
        "phase_times": trainer.phase_times,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        },
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    export_policy(trainer, path.parent / "policy.pt")
    export_policy(trainer, path.parent / f"policy_{trainer.vector_steps}.pt")
    if trainer.cfg.keep_checkpoints > 0:
        full = sorted(
            path.parent.glob("checkpoint_[0-9]*.pt"), key=lambda p: int(p.stem.split("_")[1])
        )
        for old in full[: -trainer.cfg.keep_checkpoints]:
            old.unlink()


def export_policy(trainer, path):
    # Original play.py uses load_optimizer=True even for inference. Supply a fresh
    # compatible combined optimizer schema, clearly marking this as inference-only.
    tmp = Path(path).with_suffix(".pt.tmp")
    torch.save(
        {
            "model_state_dict": trainer.policy.state_dict(),
            "iter": trainer.vector_steps // 16,
            "optimizer_state_dict": torch.optim.Adam(
                trainer.policy.parameters(), lr=5e-4
            ).state_dict(),
            "infos": {
                "algorithm": "supermimic",
                "inference_only": True,
                "vector_steps": trainer.vector_steps,
                "real_steps": trainer.vector_steps * trainer.env.num_envs,
                "descriptor": trainer.d.to_dict(),
            },
        },
        tmp,
    )
    os.replace(tmp, path)


def load_checkpoint(trainer, path):
    state = torch.load(path, map_location=trainer.device, weights_only=False)
    if state.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    if state.get("reference_sha256") != reference_hash(trainer.adapter):
        raise ValueError("Checkpoint reference clip hash differs")
    if state["descriptor"] != trainer.d.to_dict():
        raise ValueError("Checkpoint task descriptor differs")
    if state["config"] != trainer.cfg.to_dict():
        raise ValueError("Resume must use identical resolved training config")
    trainer.policy.load_state_dict(state["model_state_dict"])
    trainer.wm.load_state_dict(state["wm_state_dict"])
    for k, v in trainer.optimizers.items():
        v.load_state_dict(state["optimizers"][k])
    trainer.dual.load_state_dict(state["dual"])
    replay = state["replay"]
    example = {k: v[0].to(trainer.device) for k, v in replay["data"].items()}
    trainer.replay = ReplayBuffer(
        example,
        trainer.env.num_envs,
        trainer.cfg.replay_bytes,
        trainer.cfg.replay_max_steps,
        trainer.cfg.holdout_fraction,
        trainer.cfg.wm_window * 2,
        static_context=trainer.adapter.context,
    )
    trainer.replay.load_state_dict(replay)
    restore_environment(trainer.adapter, state["environment"])
    for k, v in state["observations"].items():
        trainer.observations[k].copy_(v)
    for k, v in state["progress"].items():
        setattr(trainer, k, v)
    trainer.elapsed, trainer.phase_times = state["elapsed"], state["phase_times"]
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"].cpu())
    if state["rng"]["cuda"]:
        torch.cuda.set_rng_state_all([v.cpu() for v in state["rng"]["cuda"]])
