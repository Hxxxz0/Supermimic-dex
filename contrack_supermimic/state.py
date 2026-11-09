from dataclasses import asdict, dataclass

import torch
from torch.nn import functional as F


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        -1,
    )


def quat_conjugate(q):
    return q * q.new_tensor([1.0, -1.0, -1.0, -1.0])


def quat_rotate_inverse(q, v):
    q = quat_conjugate(q)
    uv = torch.cross(q[..., 1:], v, dim=-1)
    return v + 2 * (q[..., :1] * uv + torch.cross(q[..., 1:], uv, dim=-1))


def rotvec_quat(v):
    angle = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    # sinc gives the correct value AND derivative at the identity rotation.
    xyz = 0.5 * torch.sinc(angle / (2 * torch.pi)) * v
    return torch.cat((torch.cos(0.5 * angle), xyz), -1)


def quaternion_error(q, r, stable=True):
    if not stable:
        return 2 * torch.acos((q * r).sum(-1).abs().clamp(0, 1))
    rel = quat_mul(q, quat_conjugate(r))
    return 2 * torch.atan2(
        torch.linalg.vector_norm(rel[..., 1:], dim=-1), rel[..., 0].abs().clamp_min(1e-8)
    )


@dataclass(frozen=True)
class TaskDescriptor:
    embodiment: str
    hand_sizes: tuple[int, ...]
    physical_joint_names: tuple[str, ...]
    object_names: tuple[str, ...]
    contact_links: tuple[tuple[str, ...], ...]
    context_dim: int
    dt: float = 1 / 60

    def __post_init__(self):
        if not self.hand_sizes or not self.object_names:
            raise ValueError("At least one hand and object are required")
        if len(self.hand_sizes) != len(self.contact_links):
            raise ValueError("One contact-link list is required per hand")
        if len({len(x) for x in self.contact_links}) != 1:
            raise ValueError("All hands must have the same tracked link count")
        if self.joints < self.actions or len(set(self.physical_joint_names)) != self.joints:
            raise ValueError("Invalid physical joint ordering")

    @property
    def hands(self):
        return len(self.hand_sizes)

    @property
    def joints(self):
        return len(self.physical_joint_names)

    @property
    def actions(self):
        return sum(self.hand_sizes)

    @property
    def objects(self):
        return len(self.object_names)

    @property
    def links(self):
        return len(self.contact_links[0])

    @property
    def contacts(self):
        return self.hands * self.objects * self.links

    @property
    def state_dim(self):
        return 2 * self.joints + 13 * self.objects

    @property
    def delta_dim(self):
        return 2 * self.joints + 12 * self.objects

    @property
    def memory_dim(self):
        return 3 + self.actions + 6 * self.objects

    @property
    def observation_dim(self):
        q = self.actions + 7 * self.objects
        v = self.actions + 6 * self.objects
        return q + v + 2 * self.contacts + 6 * q + 1

    def to_dict(self):
        return asdict(self)


class StateCodec:
    """Controlled joints first; extra physical joints next; object 13-vectors last."""

    def __init__(self, descriptor):
        self.d = descriptor

    def unpack(self, x):
        j = self.d.joints
        return (
            x[..., :j],
            x[..., j : 2 * j],
            x[..., 2 * j :].reshape(*x.shape[:-1], self.d.objects, 13),
        )

    def integrate(self, x, delta):
        q, v, obj = self.unpack(x)
        j = self.d.joints
        dobj = delta[..., 2 * j :].reshape(*delta.shape[:-1], self.d.objects, 12)
        quat = F.normalize(quat_mul(rotvec_quat(dobj[..., 3:6]), obj[..., 3:7]), dim=-1)
        nxt = torch.cat(
            (
                obj[..., :3] + dobj[..., :3],
                quat,
                obj[..., 7:10] + dobj[..., 6:9],
                obj[..., 10:13] + dobj[..., 9:12],
            ),
            -1,
        )
        return torch.cat((q + delta[..., :j], v + delta[..., j : 2 * j], nxt.flatten(-2)), -1)

    def difference(self, x1, x0):
        q1, v1, o1 = self.unpack(x1)
        q0, v0, o0 = self.unpack(x0)
        rel = quat_mul(o1[..., 3:7], quat_conjugate(o0[..., 3:7]))
        rel = torch.where(rel[..., :1] < 0, -rel, rel)
        sine = torch.linalg.vector_norm(rel[..., 1:], dim=-1, keepdim=True)
        angle = 2 * torch.atan2(sine, rel[..., :1].clamp_min(1e-8))
        rv = rel[..., 1:] * torch.where(sine > 1e-6, angle / sine.clamp_min(1e-8), 2.0)
        return torch.cat(
            (
                q1 - q0,
                v1 - v0,
                torch.cat((o1[..., :3] - o0[..., :3], rv, o1[..., 7:] - o0[..., 7:]), -1).flatten(
                    -2
                ),
            ),
            -1,
        )

    def floors(self, device=None):
        j = self.d.joints
        return torch.tensor(
            [0.003] * j
            + [0.05] * j
            + ([0.001] * 3 + [0.005] * 3 + [0.02] * 3 + [0.05] * 3) * self.d.objects,
            device=device,
        )
