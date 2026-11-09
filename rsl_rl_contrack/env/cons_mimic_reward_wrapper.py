# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from rsl_rl_contrack.env import VecEnv


class ConsMimicRewardWrapper(VecEnv):
    """Augment rsl_rl VecEnv extras with per-step task/style rewards for ConsMimic-style CMDP training.

    This wrapper computes and injects:
    - ``extras["rg"]``: task reward per env-step.
    - ``extras["rs"]``: style reward per env-step.
    - ``extras["rp"]``: penalty reward per env-step (qvel/qacc penalties), added to both task and style channels.

    Parameters
    ----------
    env : rsl_rl.env.VecEnv
        The wrapped vectorized environment. ``env.step()`` must return ``(obs, rewards, dones, extras)`` where
        ``rewards`` is a tensor of shape ``(num_envs,)``.
    isaac_env : object
        The underlying IsaacLab manager-based environment (typically ``gym_env.unwrapped``) exposing the per-step,
        reward manager buffers:
        - ``reward_manager._term_names``: list[str]
        - ``reward_manager._step_reward``: torch.Tensor, shape=(num_envs, num_terms), dtype=float32 storing
          per-term reward value divided by ``step_dt``.
        Reward groups are defined by full reward term names from ``isaac_env.cfg.cmdp_reward_groups``. The wrapper
        sums matching terms and multiplies by ``step_dt`` to match IsaacLab reward scaling.
    """

    def __init__(self, env: VecEnv, isaac_env, reward_groups: dict[str, list[str]] | None = None) -> None:
        """Initialize the reward-splitting wrapper.

        Parameters
        ----------
        env : rsl_rl.env.VecEnv
            Wrapped vectorized environment.
        isaac_env : object
            IsaacLab environment instance used to compute object tracking rewards.
        reward_groups : dict[str, list[str]] | None
            Reward group term names. When ``None``, uses ``isaac_env.cfg.cmdp_reward_groups``.
        """
        self.env = env
        self.isaac_env = isaac_env

        self.num_envs = env.num_envs
        self.num_actions = env.num_actions
        self.max_episode_length = env.max_episode_length
        self.episode_length_buf = env.episode_length_buf
        self.device = env.device
        self.cfg = env.cfg
        self._printed = False
        self.ep_rg_sum = torch.zeros(env.num_envs, device=env.device)
        self.ep_rs_sum = torch.zeros(env.num_envs, device=env.device)
        self.ep_rp_sum = torch.zeros(env.num_envs, device=env.device)
        self.it_obj_pos_err_sum = torch.zeros((), device=env.device)
        self.it_obj_rot_err_sum = torch.zeros((), device=env.device)
        self.it_step_count = 0
        self.groups = reward_groups or self.isaac_env.cfg.cmdp_reward_groups
        names = self.isaac_env.reward_manager._term_names
        name_to_id = {n: i for i, n in enumerate(names)}
        self.rg_ids = [name_to_id[n] for n in self.groups["rg"]]
        self.rp_ids = [name_to_id[n] for n in self.groups["rp"]]
        self.rs_ids = [name_to_id[n] for n in self.groups["rs"]]

    def __getattr__(self, name: str):
        """Forward attribute access to the wrapped environment.

        Parameters
        ----------
        name : str
            Attribute name.

        Returns
        -------
        object
            Attribute value from the wrapped environment.
        """
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        """Return the underlying IsaacLab environment.

        Returns
        -------
        object
            IsaacLab manager-based environment instance passed as ``isaac_env``.
        """
        return self.isaac_env

    def get_observations(self):
        """Return current observations from the wrapped environment.

        Returns
        -------
        tensordict.TensorDict
            Observation TensorDict forwarded from the wrapped env.
        """
        return self.env.get_observations()

    def step(self, actions: torch.Tensor):
        """Step the environment and add ``rg/rs/rp`` to extras.

        Parameters
        ----------
        actions : torch.Tensor, shape=(num_envs, num_actions), dtype=float32
            Action tensor in the wrapped environment's action space.

        Returns
        -------
        obs : tensordict.TensorDict
            Next observation.
        rewards : torch.Tensor, shape=(num_envs,), dtype=float32
            Total reward returned by the wrapped environment.
        dones : torch.Tensor, shape=(num_envs,), dtype=bool or uint8
            Done flags.
        extras : dict[str, torch.Tensor]
            Extra information dictionary, augmented with:
            - ``"rg"``: torch.Tensor, shape=(num_envs,), dtype=float32
            - ``"rs"``: torch.Tensor, shape=(num_envs,), dtype=float32
            - ``"rp"``: torch.Tensor, shape=(num_envs,), dtype=float32
        """
        obs, rewards, dones, extras = self.env.step(actions)
        dt = float(self.isaac_env.step_dt)
        step_reward = self.isaac_env.reward_manager._step_reward
        rg = dt * step_reward[:, self.rg_ids].sum(dim=1)
        rp = dt * step_reward[:, self.rp_ids].sum(dim=1)
        rs = dt * step_reward[:, self.rs_ids].sum(dim=1)
        objects = self.isaac_env.scene["objects"]
        idx = torch.clamp(self.isaac_env.frame_idx, max=self.isaac_env.ref_len - 1)
        pos = objects.data.object_pos_w - self.isaac_env.scene.env_origins.unsqueeze(1)
        pos_err = torch.norm(pos - self.isaac_env.ref_object_pos[idx], dim=-1)
        quat_dot = torch.abs(
            torch.clamp(
                torch.sum(
                    objects.data.object_quat_w * self.isaac_env.ref_object_quat[idx],
                    dim=-1,
                ),
                0.0,
                1.0,
            )
        )
        rot_err = 2.0 * torch.arccos(quat_dot)
        self.it_obj_pos_err_sum += pos_err.mean()
        self.it_obj_rot_err_sum += rot_err.mean()
        self.it_step_count += 1
        extras["rg"] = rg
        extras["rs"] = rs
        extras["rp"] = rp
        self.ep_rg_sum += rg
        self.ep_rs_sum += rs
        self.ep_rp_sum += rp
        done_ids = (dones > 0).nonzero(as_tuple=False)[:, 0]
        if done_ids.numel() > 0:
            extras["log"]["Episode_Reward/rg"] = self.ep_rg_sum[done_ids].mean() / float(self.isaac_env.max_episode_length_s)
            extras["log"]["Episode_Reward/rs"] = self.ep_rs_sum[done_ids].mean() / float(self.isaac_env.max_episode_length_s)
            extras["log"]["Episode_Reward/rp"] = self.ep_rp_sum[done_ids].mean() / float(self.isaac_env.max_episode_length_s)
            self.ep_rg_sum[done_ids] = 0.0
            self.ep_rs_sum[done_ids] = 0.0
            self.ep_rp_sum[done_ids] = 0.0
        if not self._printed:
            print(
                f"[ConsMimicRewardWrapper] injected extras: dt={dt} rg shape={tuple(rg.shape)} rs shape={tuple(rs.shape)} rp shape={tuple(rp.shape)} rg_mean={rg.mean().item():.4f} rs_mean={rs.mean().item():.4f} rp_mean={rp.mean().item():.4f}"
            )
            self._printed = True
        return obs, rewards, dones, extras
