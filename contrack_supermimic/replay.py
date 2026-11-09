import torch


class ReplayBuffer:
    """Time-major bounded ring with episode-aware sampling, entirely device resident."""

    def __init__(
        self,
        example,
        num_envs,
        byte_budget,
        max_steps=1000,
        holdout_fraction=0.1,
        minimum_steps=16,
        static_context=None,
    ):
        bytes_per_row = sum(v.numel() * v.element_size() for v in example.values())
        self.context = static_context
        context_bytes = (
            0 if static_context is None else static_context.numel() * static_context.element_size()
        )
        self.capacity = min(max_steps, (byte_budget - context_bytes) // bytes_per_row)
        if self.capacity < minimum_steps:
            raise ValueError(f"Replay budget permits {self.capacity} rows; need >= {minimum_steps}")
        self.device = next(iter(example.values())).device
        self.num_envs = num_envs
        self.data = {
            k: torch.empty((self.capacity, *v.shape), device=v.device, dtype=v.dtype)
            for k, v in example.items()
        }
        self.cursor = self.size = self.total_rows = 0
        count = max(1, min(num_envs - 1, round(num_envs * holdout_fraction)))
        self.holdout_envs = torch.arange(count, device=self.device)
        self.train_envs = torch.arange(count, num_envs, device=self.device)
        if num_envs < 2:
            raise ValueError("Replay requires >=2 environments")
        self.bytes = self.capacity * bytes_per_row + context_bytes

    @torch.no_grad()
    def add(self, record):
        if record.keys() != self.data.keys():
            raise ValueError("Replay schema changed")
        for key, value in record.items():
            self.data[key][self.cursor].copy_(value)
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_rows += 1

    def rows(self):
        return (
            torch.arange(self.size, device=self.device) + self.cursor - self.size
        ) % self.capacity

    def sample(self, batch, window, holdout=False, failure_fraction=0.0, failure_horizon=16):
        if self.size < window:
            raise ValueError("Insufficient replay for requested window")
        pool = self.holdout_envs if holdout else self.train_envs
        start = torch.randint(self.size - window + 1, (batch,), device=self.device)
        env = pool[torch.randint(pool.numel(), (batch,), device=self.device)]
        if failure_fraction and not holdout:
            rows = self.rows()
            fail = self.data["terminated"][rows][:, pool]
            candidates = fail.nonzero(as_tuple=False)
            n = int(batch * failure_fraction)
            if candidates.shape[0] and n:
                chosen = candidates[torch.randint(candidates.shape[0], (n,), device=self.device)]
                near = chosen[:, 0] - torch.randint(failure_horizon, (n,), device=self.device)
                start[:n] = near.clamp(0, self.size - window)
                env[:n] = pool[chosen[:, 1]]
        logical = start[:, None] + torch.arange(window, device=self.device)
        rows = (logical + self.cursor - self.size) % self.capacity
        out = {k: v[rows, env[:, None]] for k, v in self.data.items()}
        same = out["episode"] == out["episode"][:, :1]
        # Include the terminal transition itself; discard all following reset transitions.
        prior_done = torch.cat((torch.zeros_like(out["done"][:, :1]), out["done"][:, :-1]), 1)
        valid = same & (prior_done.cumsum(1) == 0) & out["valid"].cumprod(1).bool()
        out["mask"] = valid
        out["env"] = env
        if self.context is not None:
            out["context"] = self.context[env, None, :].expand(-1, window, -1)
        return out

    def starts(self, batch):
        if not self.size:
            raise ValueError("Empty replay")
        rows = self.rows()
        eligible = self.data["valid"][rows][:, self.train_envs].nonzero(as_tuple=False)
        if not eligible.shape[0]:
            raise ValueError("No finite imagination starts")
        chosen = eligible[torch.randint(eligible.shape[0], (batch,), device=self.device)]
        row, env = rows[chosen[:, 0]], self.train_envs[chosen[:, 1]]
        out = {k: v[row, env] for k, v in self.data.items()}
        if self.context is not None:
            out["context"] = self.context[env]
        return out

    def state_dict(self):
        # Serialize chronological initialized rows only, never uninitialized ring memory.
        return {
            "data": {k: v[self.rows()].cpu() for k, v in self.data.items()},
            "total_rows": self.total_rows,
            "capacity": self.capacity,
            "train_envs": self.train_envs.cpu(),
            "holdout_envs": self.holdout_envs.cpu(),
            "context": None if self.context is None else self.context.cpu(),
        }

    def load_state_dict(self, state):
        if state["capacity"] != self.capacity:
            raise ValueError("Resume replay capacity differs")
        n = next(iter(state["data"].values())).shape[0]
        for k, v in state["data"].items():
            self.data[k][:n].copy_(v)
        self.size, self.cursor = n, n % self.capacity
        self.total_rows = state["total_rows"]
        self.train_envs = state["train_envs"].to(self.device)
        self.holdout_envs = state["holdout_envs"].to(self.device)
        if state.get("context") is not None:
            if self.context is None:
                raise ValueError("Resume needs static dynamics context")
            self.context.copy_(state["context"])
