"""Read-only adapter and reward-phase capture. Upstream environment.step stays intact."""

import torch

from .state import TaskDescriptor, quat_rotate_inverse
from .task import ConTrackTask


class IsaacAdapter:
    def __init__(self, env, pos_width=0.005, rot_width=0.05):
        self.env = env
        self.device = env.device
        self.sharpa = hasattr(env, "hand_contact_sensor_keys")
        self.specs, seen = [], set()
        # The controlled prefix MUST exactly match Actor/action ordering.
        for name, ids, _ in env.ref_hands:
            for joint in ids:
                spec = (name, int(joint))
                if spec in seen:
                    raise ValueError("Duplicate controlled joint")
                self.specs.append(spec)
                seen.add(spec)
        for name in dict.fromkeys(h[0] for h in env.ref_hands):
            for joint in range(len(env.scene[name].joint_names)):
                if (name, joint) not in seen:
                    self.specs.append((name, joint))
                    seen.add((name, joint))
        self.read_groups = []
        # Consecutive groups avoid one tensor kernel per joint in real sampling.
        for name, joint in self.specs:
            if not self.read_groups or self.read_groups[-1][0] != name:
                self.read_groups.append((name, []))
            self.read_groups[-1][1].append(joint)
        self.contact_keys = []
        for name, _, hand in env.ref_hands:
            if self.sharpa:
                side = "right" if bool(env.is_rhand[hand]) else "left"
                self.contact_keys.append(tuple(env.hand_contact_sensor_keys[side]))
            else:
                self.contact_keys.append(tuple(env.xhand_contact_sensor_keys[name]))
        obj = env.scene["objects"]
        names = tuple(getattr(obj, "object_names", [str(i) for i in range(env.num_objects)]))
        self.context = self._physical_context()
        self.d = TaskDescriptor(
            "sharpa" if self.sharpa else "xhand",
            tuple(len(h[1]) for h in env.ref_hands),
            tuple(f"{name}/{env.scene[name].joint_names[j]}" for name, j in self.specs),
            names,
            tuple(self.contact_keys),
            self.context.shape[-1],
            float(env.step_dt),
        )
        hids = [h[2] for h in env.ref_hands]
        prefix = "hand" if self.sharpa else "xhand"
        refs = {
            "q": env.ref_qpos[:, hids],
            "pos": env.ref_object_pos,
            "quat": env.ref_object_quat,
            "contact": getattr(env, f"ref_{prefix}_is_contact")[:, :, hids].permute(0, 2, 1, 3),
            "points": getattr(env, f"ref_{prefix}_contact_points")[:, :, hids].permute(
                0, 2, 1, 3, 4
            ),
        }
        terms = list(env.action_manager._terms.values())
        residual = [t for t in terms if hasattr(t, "_action_scales")]
        if len(residual) != 1:
            raise ValueError("Expected one original residual action term")
        self.action_term = residual[0]
        scale = torch.cat(self.action_term._action_scales)
        self.reward_names = list(env.reward_manager._term_names)
        cfgs = env.reward_manager._term_cfgs
        weights = {n: float(c.weight) for n, c in zip(self.reward_names, cfgs)}
        params = {n: c.params for n, c in zip(self.reward_names, cfgs)}
        self.task = ConTrackTask(
            self.d,
            refs,
            env.ref_step_frames,
            scale,
            weights,
            env.cfg.cmdp_reward_groups,
            params["tracking_obj_pos"]["std"],
            params["tracking_obj_rot"]["std"],
            pos_width=pos_width,
            rot_width=rot_width,
        )
        self.groups = {
            group: torch.tensor([self.reward_names.index(n) for n in names], device=self.device)
            for group, names in env.cfg.cmdp_reward_groups.items()
        }
        self.episode = torch.zeros(env.num_envs, device=self.device, dtype=torch.long)
        self.pending = self.captured = None
        self._original_compute = None
        self._validate_observation_contract()

    def _validate_observation_contract(self):
        # The released configurations have no delay and zero noise. Reject an unsupported
        # altered contract instead of silently supplying clean observations to the Actor.
        policy = self.env.cfg.observations.policy
        for name in ("all_qpos", "all_qvel"):
            term = getattr(policy, name)
            if tuple(term.params.get("delay_range", (0, 0))) != (0, 0):
                raise ValueError("Nonzero observation delay needs an explicit history adapter")
            if term.noise and float(getattr(term.noise, "std", 0)) != 0:
                raise ValueError("This pinned reproduction expects upstream zero observation noise")
        if policy.ref_all_qpos.params.get("num_future", 5) != 5:
            raise ValueError("Expected upstream current+5 reference observations")

    def _physical_context(self):
        env = self.env
        chunks = []
        for name, ids in self.read_groups:
            data = env.scene[name].data
            for attr in (
                "joint_stiffness",
                "joint_damping",
                "joint_pos_limits",
                "joint_effort_limits",
            ):
                tensor = getattr(data, attr, None)
                if tensor is None:
                    raise ValueError(f"Missing physical parameter {name}.{attr}")
                chunks.append(tensor[:, ids].reshape(env.num_envs, -1))
            root = data.root_state_w[:, :7].clone()
            root[:, :3] -= env.scene.env_origins
            chunks.append(root)
        objects = env.scene["objects"]
        view = objects.root_physx_view
        # PhysX collection view uses an object-major flattened body layout.
        for getter in ("get_masses", "get_inertias", "get_material_properties"):
            raw = getattr(view, getter)().to(self.device)
            count = env.num_objects
            value = raw.reshape(count, env.num_envs, -1).transpose(0, 1).reshape(env.num_envs, -1)
            chunks.append(value)
        return torch.cat(chunks, -1).float()

    def state(self):
        q = torch.cat([self.env.scene[n].data.joint_pos[:, ids] for n, ids in self.read_groups], -1)
        v = torch.cat([self.env.scene[n].data.joint_vel[:, ids] for n, ids in self.read_groups], -1)
        obj = self.env.scene["objects"].data.object_state_w.clone()
        obj[..., :3] -= self.env.scene.env_origins[:, None, :]
        return torch.cat((q, v, obj.flatten(1)), -1)

    def memory(self):
        env = self.env
        arms = env.prev_arm_qvel.reshape(env.num_envs, self.d.hands, 7)
        fingers = env.prev_finger_qvel.reshape(env.num_envs, self.d.hands, -1)
        velocity = torch.cat((arms, fingers), -1).flatten(1)
        object_vel = torch.cat((env.prev_object_lin_vel, env.prev_object_ang_vel), -1).flatten(1)
        return self.task.memory(
            env.frame_idx_f,
            torch.cat((velocity, object_vel), -1),
            torch.stack((env.object_pos_threshold, env.object_rot_threshold), -1),
        )

    def contacts(self):
        env = self.env
        points_w = torch.stack(
            [
                torch.stack([env.scene[key].data.contact_pos_w[:, 0] for key in keys], -2)
                for keys in self.contact_keys
            ],
            1,
        )
        valid = torch.isfinite(points_w).all(-1)
        obj = env.scene["objects"].data
        pos = obj.object_pos_w[:, None, :, None, :]
        quat = obj.object_quat_w[:, None, :, None, :].expand(*points_w.shape[:-1], 4)
        points = quat_rotate_inverse(quat, torch.nan_to_num(points_w) - pos)
        return valid.float(), torch.where(valid[..., None], points, 0.0)

    def attach(self):
        if self._original_compute is not None:
            raise RuntimeError("Capture already attached")
        manager = self.env.reward_manager
        self._original_compute = manager.compute

        def compute(dt):
            reward = self._original_compute(dt=dt)
            if self.pending is not None:
                contact, points = self.contacts()
                record = self.pending | {
                    "x1": self.state(),
                    "contact": contact,
                    "points": points,
                    "u": self.action_term.processed_actions.clone(),
                    "action": self.action_term.raw_actions.clone(),
                    "done": self.env.reset_buf.clone().bool(),
                    "terminated": self.env.reset_terminated.clone().bool(),
                    "timeout": self.env.reset_time_outs.clone().bool(),
                    "terms": (manager._step_reward * dt).clone(),
                }
                record["channels"] = torch.stack(
                    [record["terms"][:, self.groups[g]].sum(-1) for g in ("rg", "rs", "rp")], -1
                )
                finite = torch.ones(self.env.num_envs, device=self.device, dtype=torch.bool)
                for key in ("x", "x1", "u", "memory", "context", "channels"):
                    finite &= torch.isfinite(record[key]).flatten(1).all(-1)
                record["valid"] = finite
                self.captured = record
                self.pending = None
            return reward

        manager.compute = compute

    def begin(self):
        if self.pending is not None:
            raise RuntimeError("Previous real step was not captured")
        self.captured = None
        # u is cloned in capture, after process_actions has written the actual target.
        self.pending = {
            "x": self.state(),
            "memory": self.memory(),
            "context": self.context,
            "episode": self.episode.clone(),
        }
        # capture's u must read after upstream process_actions, not before stepping.

    def finish(self):
        if self.captured is None:
            raise RuntimeError("Reward-phase capture did not run")
        out = self.captured
        self.episode.add_(out["done"].long())
        self.captured = None
        return out

    def close(self):
        if self._original_compute is not None:
            self.env.reward_manager.compute = self._original_compute
            self._original_compute = None
