from types import SimpleNamespace

import torch
from tensordict import TensorDict

from contrack_supermimic.config import TrainConfig
from contrack_supermimic.trainer import SuperMimicTrainer
from rsl_rl_contrack.modules import ActorCritic
from conftest import physical_state


def test_recursive_fit_and_actor_critic_updates(task_factory, tmp_path):
    torch.manual_seed(6)
    task = task_factory(length=100)
    d, n = task.d, 8
    x = physical_state(task, n)
    task.ref["q"].zero_()
    task.ref["pos"].zero_()
    task.ref["quat"].zero_()
    task.ref["quat"][..., 0] = 1
    task.ref["contact"].zero_()
    task.ref["points"].zero_()
    x.zero_()
    x[:, 2 * d.joints + 3] = 1
    memory = task.memory(torch.zeros(n), task.previous_velocity(x), torch.ones(n, 2))
    contact = torch.zeros(n, 1, 1, 10)
    obs = task.observation(x, memory, contact)
    observations = TensorDict({"policy": obs, "critic": obs.clone()}, batch_size=[n])
    env = SimpleNamespace(num_envs=n, device="cpu", get_observations=lambda: observations)
    adapter = SimpleNamespace(task=task, d=d, attach=lambda: None)
    cfg = TrainConfig.from_preset(
        "e1",
        budget_updates=20,
        warmup_steps=16,
        wm_only_updates=1,
        wm_hidden=(32, 32),
        wm_batch=16,
        wm_microbatch=8,
        wm_window=4,
        horizon=4,
        actor_batch=16,
        actor_microbatch=8,
        critic_epochs=1,
        critic_minibatches=2,
        compile=False,
        lr=0.002,
    )
    policy = ActorCritic(
        observations,
        {"policy": ["policy"], "critic": ["critic"]},
        d.actions,
        actor_hidden_dims=[16],
        critic_hidden_dims=[16],
    )
    trainer = SuperMimicTrainer(env, adapter, policy, cfg, tmp_path)
    from contrack_supermimic.replay import ReplayBuffer

    context = torch.zeros(n, d.context_dim)
    for t in range(40):
        u = torch.randn(n, d.actions) * 0.03
        delta = torch.zeros(n, d.delta_dim)
        delta[:, : d.joints] = 0.1 * (u - x[:, : d.joints])
        delta[:, 2 * d.joints] = u[:, 0] * 0.02
        x1 = task.codec.integrate(x, delta)
        record = {
            "x": x,
            "x1": x1,
            "u": u,
            "memory": memory,
            "initial_contact": contact,
            "contact": contact,
            "points": torch.zeros(n, 1, 1, 10, 3),
            "episode": torch.zeros(n, dtype=torch.long),
            "done": torch.zeros(n, dtype=torch.bool),
            "terminated": torch.zeros(n, dtype=torch.bool),
            "valid": torch.ones(n, dtype=torch.bool),
        }
        if trainer.replay is None:
            trainer.replay = ReplayBuffer(
                record, n, 1_000_000, max_steps=64, minimum_steps=16, static_context=context
            )
        trainer.wm.observe(x, u, context, x1)
        trainer.replay.add(record)
        x, memory = x1, task.advance(x1, memory)
    batch = trainer.replay.sample(16, 4)
    before = trainer._wm_loss(batch)[0].item()
    for _ in range(200):
        trainer.fit_wm()
    after = trainer._wm_loss(batch)[0].item()
    assert after < before * 0.8
    changed = {k: v.clone() for k, v in policy.actor.state_dict().items()}
    wm_before = {k: v.clone() for k, v in trainer.wm.state_dict().items()}
    trainer.policy_update()
    assert any(not torch.equal(changed[k], v) for k, v in policy.actor.state_dict().items())
    for k, v in trainer.wm.state_dict().items():
        torch.testing.assert_close(v, wm_before[k])
    # Later teacher-forced states must be ignored by recursive training.
    altered = {k: v.clone() for k, v in batch.items()}
    altered["x"][:, 1:] += 100
    torch.testing.assert_close(trainer._wm_loss(batch)[0], trainer._wm_loss(altered)[0])
    # Gradient accumulation must preserve global masking, even if failure-heavy
    # sequences have unequal valid lengths across microbatches.
    batch["mask"][:8, 1:] = False
    loss, parts = trainer._wm_loss(batch)
    expected = (
        cfg.final_state_weight * parts["state_loss"]
        + cfg.contact_weight * parts["contact_loss"]
        + cfg.point_weight * parts["point_loss"]
    )
    torch.testing.assert_close(
        loss.detach(), expected * batch["mask"].sum() / batch["mask"].shape[0]
    )
    trainer.optimizers["wm"].zero_grad(set_to_none=True)
    trainer._wm_loss(batch)[0].backward()
    full = [p.grad.clone() if p.grad is not None else None for p in trainer.wm.parameters()]
    trainer.optimizers["wm"].zero_grad(set_to_none=True)
    denom = batch["mask"].float().sum()
    for lo in (0, 8):
        micro = {k: v[lo : lo + 8] for k, v in batch.items()}
        trainer._wm_loss(micro, denom)[0].backward()
    for expected, p in zip(full, trainer.wm.parameters()):
        if expected is not None:
            torch.testing.assert_close(p.grad, expected, atol=3e-6, rtol=3e-5)
