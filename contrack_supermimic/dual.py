from collections import deque
import math

import torch


class DualController:
    """Original real episodic rg controller, independent of imagination update count."""

    def __init__(
        self, num_envs, device, alpha=0.9, lr=0.05, initial=0.0, window=100, star_window=20
    ):
        self.alpha, self.lr, self.initial = alpha, lr, initial
        self.raw = initial
        self.returns = deque(maxlen=window)
        self.estimates = deque(maxlen=star_window)
        self.episode_sum = torch.zeros(num_envs, device=device)
        self.vector_steps = self.updates = 0
        self.hat = self.star = 0.0

    @property
    def weight(self):
        return 1 / (1 + math.exp(-max(-80.0, min(80.0, self.raw))))

    @torch.no_grad()
    def observe(self, rg, done):
        self.episode_sum.add_(rg)
        ids = done.nonzero(as_tuple=False).flatten()
        if ids.numel():
            self.returns.extend(self.episode_sum[ids].cpu().tolist())
            self.episode_sum[ids] = 0
        self.vector_steps += 1
        if self.vector_steps % 16 == 0:
            self.updates += 1
            if self.returns:
                self.hat = sum(self.returns) / len(self.returns)
                self.estimates.append(self.hat)
                self.star = max(self.estimates)
                self.raw += self.lr * (self.alpha - self.hat / (self.star + 1e-8))

    def state_dict(self):
        return {
            "raw": self.raw,
            "returns": list(self.returns),
            "estimates": list(self.estimates),
            "episode_sum": self.episode_sum.cpu(),
            "vector_steps": self.vector_steps,
            "updates": self.updates,
            "hat": self.hat,
            "star": self.star,
        }

    def load_state_dict(self, s):
        for k in ("raw", "vector_steps", "updates", "hat", "star"):
            setattr(self, k, s[k])
        self.returns.extend(s["returns"])
        self.estimates.extend(s["estimates"])
        self.episode_sum.copy_(s["episode_sum"])
