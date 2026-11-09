from contextlib import contextmanager

import torch


@contextmanager
def freeze_parameters(*modules):
    saved = [(p, p.requires_grad) for module in modules for p in module.parameters()]
    try:
        for p, _ in saved:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in saved:
            p.requires_grad_(flag)


def lambda_returns(
    rewards, values, continuation, timeout, active, gamma=0.99, lam=0.95, next_values=None
):
    """Twin-channel returns; timeouts bootstrap, physical failures remove continuation."""
    ret = values[-1]
    out = []
    for t in reversed(range(rewards.shape[0])):
        vnext = values[t + 1] if next_values is None else next_values[t]
        next_return = torch.where(timeout[t, :, None], vnext, (1 - lam) * vnext + lam * ret)
        target = rewards[t] + gamma * continuation[t, :, None] * next_return
        ret = torch.where(active[t, :, None], target, values[t])
        out.append(ret)
    return torch.stack(out[::-1])


def imagine(task, wm_step, actor, critic, start, horizon, gamma, weight):
    """Donor-style mean-action full BPTT, with no detach in the recurrent state chain."""
    x, memory = start["x"], start["memory"]
    probability = start["initial_contact"]
    survival = x.new_ones(x.shape[0])
    active = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    objective = x.new_zeros(x.shape[0])
    observations, next_observations, rewards, continues, timeouts, masks = [], [], [], [], [], []
    mix = x.new_tensor([weight, 1 - weight])
    for t in range(horizon):
        obs = task.observation(x, memory, probability)
        action = actor(obs)
        u = task.control(action, memory)
        x1, logits, points1 = wm_step(x, u, start["context"])
        probability1 = logits.sigmoid()
        channels, _ = task.rewards(x1, memory, probability1, points1, soft=True)
        twin_reward = channels[:, :2] + channels[:, 2:3]
        cont = task.continuation(x1, memory)
        timeout = task.timeout(memory) & active
        next_memory = task.advance(x1, memory)
        next_obs = task.observation(x1, next_memory, probability1)
        objective = objective + gamma**t * survival * (twin_reward * mix).sum(-1) * active
        next_survival = survival * cont
        # At a reference timeout bootstrap here, rather than imagining a reset/new clip.
        bootstrap = (critic(next_obs) * mix).sum(-1)
        objective = objective + gamma ** (t + 1) * next_survival * bootstrap * timeout
        observations.append(obs.detach())
        next_observations.append(next_obs.detach())
        rewards.append(twin_reward.detach())
        continues.append(cont.detach())
        timeouts.append(timeout)
        masks.append(active)
        survival = torch.where(active, next_survival, survival)
        active = active & ~timeout
        # Keep inactive samples finite and stationary while using fixed compiled shapes.
        x = torch.where(active[:, None], x1, x)
        memory = torch.where(active[:, None], next_memory, memory)
        probability = torch.where(active[:, None, None, None], probability1, probability)
        # Critic's next value at timeout must be the terminal physical observation.
    final_obs = task.observation(x, memory, probability)
    objective = objective + gamma**horizon * survival * (critic(final_obs) * mix).sum(-1) * active
    # Store per-transition next observations explicitly: at timeout x/memory above stay
    # stationary and must not overwrite the terminal bootstrap observation.
    return -objective.mean(), {
        "obs": torch.stack(observations),
        "next_obs": torch.stack(next_observations),
        "rewards": torch.stack(rewards),
        "continuation": torch.stack(continues),
        "timeout": torch.stack(timeouts),
        "active": torch.stack(masks),
        "final_obs": final_obs.detach(),
    }
