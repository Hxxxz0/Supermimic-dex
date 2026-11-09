"""Pure tensor versions of ConTrack task functions; no simulator calls in imagination."""

import torch

from .state import StateCodec, quaternion_error


class ConTrackTask:
    def __init__(
        self,
        descriptor,
        references,
        frame_step,
        action_scale,
        weights,
        groups,
        pos_std=0.05,
        rot_std=0.5,
        contact_std=0.03,
        pos_width=0.005,
        rot_width=0.05,
        velocity_cache_updates=False,
    ):
        self.d = descriptor
        self.codec = StateCodec(descriptor)
        self.ref = references  # q [T,H,Jh]; contacts [T,H,O,L]; points [T,H,O,L,3]
        self.length = references["q"].shape[0]
        self.frame_step = float(frame_step)
        self.scale = action_scale
        self.weights = weights
        self.groups = groups
        self.pos_std, self.rot_std, self.contact_std = pos_std, rot_std, contact_std
        self.pos_width, self.rot_width = pos_width, rot_width
        # Pinned upstream uses tensor[advanced_indices].copy_(), which writes a copy.
        # Its velocity caches therefore stay at reset values. Preserve actual rewards.
        self.velocity_cache_updates = velocity_cache_updates
        self.future = torch.arange(6, device=action_scale.device) * self.frame_step
        arm, finger = [], []
        offset = 0
        for size in descriptor.hand_sizes:
            arm.extend(range(offset, offset + 7))
            finger.extend(range(offset + 7, offset + size))
            offset += size
        self.arm = torch.tensor(arm, device=action_scale.device)
        self.finger = torch.tensor(finger, device=action_scale.device)

    def memory(self, frame, previous_vel, thresholds):
        return torch.cat((frame[:, None], previous_vel, thresholds), -1)

    def index(self, memory):
        return memory[..., 0].long().clamp(0, self.length - 1)

    def reference_q(self, memory):
        return self.ref["q"][self.index(memory)].flatten(-2)

    def control(self, action, memory):
        return self.reference_q(memory) + action.clamp(-1, 1) * self.scale

    def qpos_qvel(self, x):
        q, v, obj = self.codec.unpack(x)
        return (
            torch.cat((q[..., : self.d.actions], obj[..., :7].flatten(-2)), -1),
            torch.cat((v[..., : self.d.actions], obj[..., 7:].flatten(-2)), -1),
        )

    def previous_velocity(self, x):
        _, v, obj = self.codec.unpack(x)
        return torch.cat((v[..., : self.d.actions], obj[..., 7:].flatten(-2)), -1)

    def advance(self, x1, memory):
        frame = (memory[..., :1] + self.frame_step).clamp(max=self.length - 1)
        velocity = self.previous_velocity(x1) if self.velocity_cache_updates else memory[..., 1:-2]
        return torch.cat((frame, velocity, memory[..., -2:]), -1)

    def observation(self, x, memory, contact_probability, straight_through=True):
        qpos, qvel = self.qpos_qvel(x)
        idx = self.index(memory)
        future_idx = (memory[..., :1] + self.future).long().clamp(0, self.length - 1)
        fq = self.ref["q"][future_idx].flatten(-2)
        fo = torch.cat((self.ref["pos"][future_idx], self.ref["quat"][future_idx]), -1).flatten(-2)
        future = torch.cat((fq, fo), -1).flatten(-2)
        ref_contact = self.ref["contact"][idx].flatten(-3).float()
        hard = (contact_probability >= 0.5).float()
        contact = (
            hard + contact_probability - contact_probability.detach() if straight_through else hard
        )
        phase = idx.float().unsqueeze(-1) / (self.length - 1)
        return torch.cat((qpos, qvel, contact.flatten(-3), future, ref_contact, phase), -1)

    def pose_error(self, x, memory, stable=True):
        _, _, obj = self.codec.unpack(x)
        idx = self.index(memory)
        pos = torch.linalg.vector_norm(obj[..., :3] - self.ref["pos"][idx], dim=-1)
        rot = quaternion_error(obj[..., 3:7], self.ref["quat"][idx], stable=stable)
        return pos, rot

    def continuation(self, x1, memory):
        pos, rot = self.pose_error(x1, memory)
        good_pos = torch.sigmoid((memory[..., -2:-1] - pos) / self.pos_width)
        good_rot = torch.sigmoid((memory[..., -1:] - rot) / self.rot_width)
        return (good_pos * good_rot).prod(-1)

    def timeout(self, memory):
        return self.index(memory) >= self.length - 1

    def rewards(self, x1, memory, probability, points, soft=True):
        q, v, obj = self.codec.unpack(x1)
        idx = self.index(memory)
        eq = q[..., : self.d.actions] - self.reference_q(memory)
        terms = {
            "tracking_arm": torch.exp(-4 * eq[..., self.arm].square().mean(-1)),
            "tracking_finger": torch.exp(-4 * eq[..., self.finger].square().mean(-1)),
            "tracking_obj_pos": torch.exp(
                -(obj[..., :3] - self.ref["pos"][idx]).square().mean((-1, -2)) / self.pos_std**2
            ),
            # Upstream squares the mean angle, NOT the mean squared angle.
            "tracking_obj_rot": torch.exp(
                -quaternion_error(obj[..., 3:7], self.ref["quat"][idx], stable=soft)
                .mean(-1)
                .square()
                / self.rot_std**2
            ),
        }
        velocity = self.previous_velocity(x1)
        old = memory[..., 1:-2]
        acceleration = (velocity - old) / self.d.dt
        groups = {
            "arm": self.arm,
            "finger": self.finger,
            "obj_pos": torch.arange(self.d.actions, velocity.shape[-1], device=x1.device)
            .reshape(-1, 6)[:, :3]
            .flatten(),
            "obj_rot": torch.arange(self.d.actions, velocity.shape[-1], device=x1.device)
            .reshape(-1, 6)[:, 3:]
            .flatten(),
        }
        for name, ids in groups.items():
            terms["qvel_penalty_" + name] = -torch.expm1(-velocity[..., ids].square().mean(-1))
            terms["qacc_penalty_" + name] = -torch.expm1(
                -acceleration[..., ids].square().mean(-1) / 50**2
            )
        ref_c = self.ref["contact"][idx].float()
        p = probability if soft else (probability >= 0.5).float()
        match = ref_c * p
        link_match = 1 - (1 - match).prod(-2)
        terms["contact_reward"] = link_match.mean((-1, -2))
        err = (points - self.ref["points"][idx]).square().sum(-1)
        count = match.sum(-2)
        # Normalized point error: in the binary limit this is upstream's masked object mean.
        mse = (err * match).sum(-2) / count.clamp_min(1e-8 if soft else 1)
        valid_link = link_match if soft else (count > 0).float()
        point_reward = torch.exp(-mse / self.contact_std**2) * valid_link
        per_hand = point_reward.sum(-1) / valid_link.sum(-1).clamp_min(1e-8 if soft else 1)
        if soft:
            # Upstream also gates the hand's conditional mean by (cnt > 0).
            # Without this soft OR, arbitrarily tiny contact probabilities can
            # produce a full contact-distance reward after normalization.
            per_hand = per_hand * (1 - (1 - valid_link).prod(-1))
        terms["contact_distance_reward"] = per_hand.mean(-1)
        weighted = {name: value * self.weights[name] * self.d.dt for name, value in terms.items()}
        channels = torch.stack(
            [sum(weighted[name] for name in self.groups[group]) for group in ("rg", "rs", "rp")], -1
        )
        return channels, weighted
