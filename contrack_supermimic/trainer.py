import json
import signal
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .bptt import freeze_parameters, imagine, lambda_returns
from .dual import DualController
from .model import WorldModel
from .replay import ReplayBuffer


class SuperMimicTrainer:
    def __init__(self, env, adapter, policy, cfg, run_dir):
        self.env, self.adapter, self.policy, self.cfg = env, adapter, policy, cfg
        self.task, self.d = adapter.task, adapter.d
        self.device = torch.device(env.device)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.wm = WorldModel(self.d, cfg.wm_hidden, cfg.amp).to(self.device)
        if hasattr(policy, "std"):
            policy.std.requires_grad_(False)
        if hasattr(policy, "log_std"):
            policy.log_std.requires_grad_(False)
        self.optimizers = {
            "wm": torch.optim.Adam(self.wm.parameters(), lr=cfg.lr),
            "actor": torch.optim.Adam(policy.actor.parameters(), lr=cfg.lr),
            "critic": torch.optim.Adam(policy.critic.parameters(), lr=cfg.lr),
        }
        self.dual = DualController(env.num_envs, self.device)
        self.replay = None
        self.vector_steps = self.iteration = self.policy_updates = 0
        self.collect_accumulator = 0.0
        self.elapsed = 0.0
        self.phase_times = {k: 0.0 for k in ("collect", "wm", "actor", "critic", "checkpoint")}
        self.pending_events = []
        self.stop_requested = False
        self.observations = env.get_observations()
        self.actor_net = policy.actor
        self.critic_net = policy.critic
        self.wm_step = self.wm
        if cfg.compile and self.device.type == "cuda":
            self.wm_step = torch.compile(self.wm)
            self.actor_net = torch.compile(policy.actor)
            self.critic_net = torch.compile(policy.critic)
            # Compilation of pure task tensor kernels avoids a giant unrolled graph.
            self.task.observation = torch.compile(self.task.observation)
            self.task.rewards = torch.compile(self.task.rewards)
        self.adapter.attach()
        self.last_log = self.last_save = -1
        self.parity_max = {}
        self.parity_count = 0
        self.latest = {}
        self.real_reward_sum = torch.zeros(3, device=self.device)
        self.real_episode_count = torch.zeros((), device=self.device)
        self.real_completion_count = torch.zeros((), device=self.device)
        self.real_stat_steps = 0
        self.last_real_statistics = {}

    def actor(self, obs):
        return torch.tanh(self.actor_net(self.policy.actor_obs_normalizer(obs)))

    def critic(self, obs):
        return self.critic_net(self.policy.critic_obs_normalizer(obs))

    def phase(self, name):
        trainer = self

        class Timer:
            def __enter__(self):
                self.wall = time.perf_counter()
                if trainer.device.type == "cuda":
                    self.start, self.end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    self.start.record()
                return self

            def __exit__(self, *args):
                if trainer.device.type == "cuda":
                    self.end.record()
                    trainer.pending_events.append((name, self.start, self.end))
                else:
                    trainer.phase_times[name] += time.perf_counter() - self.wall

        return Timer()

    @torch.no_grad()
    def collect(self, steps):
        with self.phase("collect"):
            for _ in range(steps):
                self.policy.update_normalization(self.observations)
                action = self.policy.act(self.observations)
                initial_contact, _ = self.adapter.contacts()
                self.adapter.begin()
                obs, _, _, _ = self.env.step(action)
                record = self.adapter.finish()
                record.update(initial_contact=initial_contact)
                stored = {
                    k: v for k, v in record.items() if k not in ("context", "terms", "action")
                }
                if self.replay is None:
                    self.replay = ReplayBuffer(
                        stored,
                        self.env.num_envs,
                        self.cfg.replay_bytes,
                        self.cfg.replay_max_steps,
                        self.cfg.holdout_fraction,
                        minimum_steps=self.cfg.wm_window * 2,
                        static_context=self.adapter.context,
                    )
                ids = self.replay.train_envs
                finite_ids = ids[record["valid"][ids]]
                self.wm.observe(
                    record["x"][finite_ids],
                    record["u"][finite_ids],
                    record["context"][finite_ids],
                    record["x1"][finite_ids],
                )
                self.replay.add(stored)
                self.dual.observe(record["channels"][:, 0], record["done"])
                self.real_reward_sum += record["channels"].mean(0)
                self.real_episode_count += record["done"].sum()
                self.real_completion_count += (record["timeout"] & ~record["terminated"]).sum()
                self.real_stat_steps += 1
                if self.parity_count < self.cfg.parity_steps:
                    self.check_parity(record)
                self.observations = obs
                self.vector_steps += 1

    @torch.no_grad()
    def check_parity(self, record):
        _, terms = self.task.rewards(
            record["x1"], record["memory"], record["contact"], record["points"], soft=False
        )
        for index, name in enumerate(self.adapter.reward_names):
            gap = (terms[name] - record["terms"][:, index]).abs()
            selected = gap[record["valid"]]
            if selected.numel():
                maximum = selected.max().item()
                self.parity_max[name] = max(self.parity_max.get(name, 0.0), maximum)
                tolerance = 2e-4 + 2e-4 * record["terms"][:, index].abs().max().item()
                if maximum > tolerance:
                    raise RuntimeError(f"Reward parity failed for {name}: {maximum} > {tolerance}")
        reconstructed = self.task.observation(
            record["x"], record["memory"], record["initial_contact"], False
        )
        source = self.observations["policy"]
        if reconstructed.shape != source.shape:
            raise RuntimeError("Actor observation dimension changed")
        gap = (reconstructed - source).abs().max().item()
        self.parity_max["observation"] = max(self.parity_max.get("observation", 0.0), gap)
        if gap > 2e-4:
            raise RuntimeError(f"Actor observation parity failed: {gap}")
        expected_u = self.task.control(record["action"], record["memory"])
        if not torch.allclose(expected_u, record["u"], atol=1e-6, rtol=1e-6):
            raise RuntimeError("Executed target differs from imagined action mapping")
        self.parity_count += 1

    def _wm_loss(self, batch, denominator=None):
        x = torch.nan_to_num(batch["x"][:, 0])
        mask = batch["mask"].float()
        state_losses, contact_losses, point_losses = [], [], []
        tp = fp = fn = torch.zeros((), device=self.device)
        scale = torch.maximum(self.wm.deltas.std(), self.wm.error_floor)
        for t in range(self.cfg.wm_window):
            x, logits, points = self.wm_step(
                x, torch.nan_to_num(batch["u"][:, t]), torch.nan_to_num(batch["context"][:, t])
            )
            target = torch.nan_to_num(batch["x1"][:, t])
            error = self.wm.codec.difference(x, target) / scale
            state_losses.append(error.square().mean(-1))
            contact_losses.append(
                F.binary_cross_entropy_with_logits(logits, batch["contact"][:, t], reduction="none")
                .flatten(1)
                .sum(-1)
            )
            cmask = batch["contact"][:, t]
            point_error = (points - batch["points"][:, t]).square().sum(-1) / 0.03**2
            point_losses.append(
                (point_error * cmask).flatten(1).sum(-1) / cmask.flatten(1).sum(-1).clamp_min(1)
            )
            pred = logits >= 0
            truth = cmask.bool()
            valid = mask[:, t, None, None, None]
            tp = tp + ((pred & truth) * valid).sum()
            fp = fp + ((pred & ~truth) * valid).sum()
            fn = fn + ((~pred & truth) * valid).sum()
        states, contacts, points = (
            torch.stack(v, 1) for v in (state_losses, contact_losses, point_losses)
        )
        denom = mask.sum().clamp_min(1) if denominator is None else denominator
        sl = (states * mask).sum() / denom
        cl = (contacts * mask).sum() / denom
        pl = (points * mask).sum() / denom
        # Donor's final_state_weight multiplies every recursively predicted state.
        # Donor optimizes the masked temporal SUM per sampled sequence; reported
        # components are means per valid transition. Preserve that loss scale
        # across microbatches, including sequences truncated by episode ends.
        batch_size = mask.shape[0] if denominator is None else self.cfg.wm_batch
        loss = (
            (
                self.cfg.final_state_weight * sl
                + self.cfg.contact_weight * cl
                + self.cfg.point_weight * pl
            )
            * denom
            / batch_size
        )
        return loss, {
            "state_loss": sl.detach(),
            "contact_loss": cl.detach(),
            "point_loss": pl.detach(),
            "contact_tp": tp.detach(),
            "contact_fp": fp.detach(),
            "contact_fn": fn.detach(),
        }

    def optimize(self, name, parameters):
        params = list(parameters)
        norm = torch.nn.utils.clip_grad_norm_(
            params, self.cfg.max_grad_norm, error_if_nonfinite=True
        )
        if name == "actor":
            for p in params:
                if p.grad is not None:
                    p.grad.clamp_(-self.cfg.actor_grad_clip, self.cfg.actor_grad_clip)
        self.optimizers[name].step()
        return norm.detach()

    def fit_wm(self, holdout=False):
        batch = self.replay.sample(
            self.cfg.wm_batch,
            self.cfg.wm_window,
            holdout,
            0.0 if holdout else self.cfg.failure_fraction,
            self.cfg.failure_horizon,
        )
        stats = {}
        if not holdout:
            self.optimizers["wm"].zero_grad(set_to_none=True)
        denominator = batch["mask"].float().sum().clamp_min(1)
        for start in range(0, self.cfg.wm_batch, self.cfg.wm_microbatch):
            end = min(start + self.cfg.wm_microbatch, self.cfg.wm_batch)
            micro = {k: v[start:end] for k, v in batch.items()}
            with torch.set_grad_enabled(not holdout):
                loss, part = self._wm_loss(micro, denominator)
            if not holdout:
                loss.backward()
            for k, v in part.items():
                stats[k] = stats.get(k, 0) + v
        if not holdout:
            stats["grad_norm"] = self.optimize("wm", self.wm.parameters())
        stats["contact_f1"] = (
            2
            * stats["contact_tp"]
            / (2 * stats["contact_tp"] + stats["contact_fp"] + stats["contact_fn"]).clamp_min(1)
        )
        return stats

    def policy_update(self):
        starts = self.replay.starts(self.cfg.actor_batch)
        trajectories = []
        self.optimizers["actor"].zero_grad(set_to_none=True)
        self.optimizers["wm"].zero_grad(set_to_none=True)
        self.optimizers["critic"].zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=self.device)
        with self.phase("actor"), freeze_parameters(self.wm, self.policy.critic):
            for lo in range(0, self.cfg.actor_batch, self.cfg.actor_microbatch):
                hi = min(lo + self.cfg.actor_microbatch, self.cfg.actor_batch)
                loss, trajectory = imagine(
                    self.task,
                    self.wm_step,
                    self.actor,
                    self.critic,
                    {k: v[lo:hi] for k, v in starts.items()},
                    self.cfg.horizon,
                    self.cfg.gamma,
                    self.dual.weight,
                )
                fraction = (hi - lo) / self.cfg.actor_batch
                (loss * fraction).backward()
                total_loss += loss.detach() * fraction
                trajectories.append(trajectory)
            grad = self.optimize("actor", self.policy.actor.parameters())
        data = {
            k: torch.cat([v[k] for v in trajectories], dim=0 if k == "final_obs" else 1)
            for k in trajectories[0]
        }
        with self.phase("critic"):
            obs = data["obs"]
            with torch.no_grad():
                values = torch.cat((self.critic(obs), self.critic(data["final_obs"])[None]), 0)
                targets = lambda_returns(
                    data["rewards"],
                    values,
                    data["continuation"],
                    data["timeout"],
                    data["active"],
                    self.cfg.gamma,
                    self.cfg.lam,
                    next_values=self.critic(data["next_obs"]),
                )
            valid = data["active"].flatten()
            inputs = obs.flatten(0, 1)[valid]
            targets = targets.flatten(0, 1)[valid]
            critic_loss = torch.zeros((), device=self.device)
            for _ in range(self.cfg.critic_epochs):
                permutation = torch.randperm(inputs.shape[0], device=self.device)
                for ids in permutation.tensor_split(self.cfg.critic_minibatches):
                    if not ids.numel():
                        continue
                    self.optimizers["critic"].zero_grad(set_to_none=True)
                    critic_loss = (self.critic(inputs[ids]) - targets[ids]).square().mean()
                    critic_loss.backward()
                    self.optimize("critic", self.policy.critic.parameters())
        self.policy_updates += 1
        return {
            "actor_loss": total_loss,
            "critic_loss": critic_loss.detach(),
            "actor_grad_norm": grad,
        }

    def learn(self):
        self.started = time.perf_counter()

        def stop(*_):
            self.stop_requested = True

        old_signals = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            if self.vector_steps < self.cfg.warmup_steps:
                self.collect(self.cfg.warmup_steps - self.vector_steps)
            while self.vector_steps < self.cfg.budget_updates * 16 and not self.stop_requested:
                warming = self.iteration < self.cfg.wm_only_updates
                self.collect_accumulator += 1 / (self.cfg.warmup_utd if warming else self.cfg.utd)
                steps = int(self.collect_accumulator)
                self.collect_accumulator -= steps
                self.collect(min(steps, self.cfg.budget_updates * 16 - self.vector_steps))
                with self.phase("wm"):
                    for _ in range(self.cfg.warmup_wm_batches if warming else self.cfg.wm_batches):
                        self.latest.update({"wm/" + k: v for k, v in self.fit_wm().items()})
                if not warming:
                    self.latest.update(self.policy_update())
                self.iteration += 1
                equivalent = self.vector_steps // 16
                if equivalent // self.cfg.log_interval > self.last_log:
                    self.latest.update({"holdout/" + k: v for k, v in self.fit_wm(True).items()})
                    self.log()
                    self.last_log = equivalent // self.cfg.log_interval
                if equivalent // self.cfg.save_interval > self.last_save:
                    self.save(self.run_dir / f"checkpoint_{self.vector_steps}.pt")
                    self.last_save = equivalent // self.cfg.save_interval
            self.log()
            self.save(self.run_dir / "checkpoint_final.pt")
        finally:
            self.elapsed += time.perf_counter() - self.started
            self.adapter.close()
            for s, handler in old_signals.items():
                signal.signal(s, handler)

    def log(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            for name, start, end in self.pending_events:
                self.phase_times[name] += start.elapsed_time(end) / 1000
            self.pending_events.clear()
        elapsed = self.elapsed + time.perf_counter() - self.started
        progress = self.vector_steps / (16 * self.cfg.budget_updates)
        row = {
            "iteration": self.iteration,
            "policy_updates": self.policy_updates,
            "vector_steps": self.vector_steps,
            "real_steps": self.vector_steps * self.env.num_envs,
            "equivalent_ppo_iteration": self.vector_steps / 16,
            "wall_seconds": elapsed,
            "estimated_remaining_seconds": elapsed * (1 - progress) / max(progress, 1e-8),
            "task_weight": self.dual.weight,
            "vg_hat": self.dual.hat,
            "vg_star": self.dual.star,
            "phase_seconds": self.phase_times.copy(),
            "parity_max": self.parity_max.copy(),
            **{k: float(v) for k, v in self.latest.items()},
        }
        if self.real_stat_steps:
            reward = (self.real_reward_sum / self.real_stat_steps).tolist()
            self.last_real_statistics = dict(
                rg=reward[0],
                rs=reward[1],
                rp=reward[2],
                training_clip_completion_fraction=float(
                    self.real_completion_count / self.real_episode_count.clamp_min(1)
                ),
            )
        row.update(
            self.last_real_statistics,
            replay_bytes=self.replay.bytes if self.replay is not None else 0,
        )
        self.real_reward_sum.zero_()
        self.real_episode_count.zero_()
        self.real_completion_count.zero_()
        self.real_stat_steps = 0
        with (self.run_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    def save(self, path):
        from .checkpoint import save_checkpoint

        with self.phase("checkpoint"):
            save_checkpoint(self, path)

    def load(self, path):
        from .checkpoint import load_checkpoint

        load_checkpoint(self, path)
