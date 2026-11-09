from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TrainConfig:
    preset: str = "e1"
    seed: int = 0
    budget_updates: int = 5000  # equivalent PPO updates, each 16 vector steps
    warmup_steps: int = 64
    wm_only_updates: int = 500
    warmup_utd: float = 0.4
    warmup_wm_batches: int = 10
    utd: float = 1 / 16
    wm_batches: int = 2
    wm_batch: int = 2048
    wm_microbatch: int = 512
    wm_window: int = 8
    wm_hidden: tuple[int, ...] = (2048, 2048, 2048)
    failure_fraction: float = 0.25
    failure_horizon: int = 16
    holdout_fraction: float = 0.1
    contact_weight: float = 0.1
    point_weight: float = 0.1
    final_state_weight: float = 1.0
    horizon: int = 32
    actor_batch: int = 1024
    actor_microbatch: int = 256
    critic_epochs: int = 5
    critic_minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    lr: float = 2e-4
    max_grad_norm: float = 1.0
    actor_grad_clip: float = 0.1
    pos_soft_width: float = 0.005
    rot_soft_width: float = 0.05
    replay_bytes: int = 2 * 1024**3
    replay_max_steps: int = 1000
    amp: bool = True
    compile: bool = True
    log_interval: int = 16  # equivalent PPO updates
    save_interval: int = 200
    keep_checkpoints: int = 2
    parity_steps: int = 16

    def __post_init__(self):
        for key in (
            "budget_updates",
            "warmup_steps",
            "wm_only_updates",
            "wm_batches",
            "wm_batch",
            "wm_microbatch",
            "wm_window",
            "horizon",
            "actor_batch",
            "actor_microbatch",
            "critic_epochs",
            "critic_minibatches",
            "replay_bytes",
            "replay_max_steps",
            "log_interval",
            "save_interval",
        ):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if not 0 < self.utd <= 1 or not 0 < self.warmup_utd <= 1:
            raise ValueError("UTD must be in (0, 1]")
        if not 0 < self.holdout_fraction < 1 or not 0 <= self.failure_fraction < 1:
            raise ValueError("Invalid replay sampling fractions")
        if not 0 < self.gamma < 1 or not 0 <= self.lam <= 1:
            raise ValueError("Invalid discount or lambda")
        if self.warmup_steps >= self.budget_updates * 16:
            raise ValueError("Real-step budget must exceed initial collection")

    @classmethod
    def from_preset(cls, preset="e1", **kwargs):
        presets = {"e1": (2, 1 / 16), "e2": (5, 1 / 16), "e3": (10, 1 / 16), "reference": (10, 0.4)}
        if preset not in presets:
            raise ValueError(f"Unknown preset {preset}")
        batches, utd = presets[preset]
        return cls(**({"preset": preset, "wm_batches": batches, "utd": utd} | kwargs))

    def to_dict(self):
        return asdict(self)
