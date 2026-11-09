from types import SimpleNamespace

import torch

from contrack_supermimic.bptt import freeze_parameters, imagine, lambda_returns
from contrack_supermimic.dual import DualController
from contrack_supermimic.replay import ReplayBuffer
from contrack_supermimic.isaac_adapter import IsaacAdapter


def test_bptt_finite_difference_through_frozen_dynamics():
    class Task:
        def observation(self, x, m, p):
            return x

        def control(self, a, m):
            return a

        def rewards(self, x, m, p, points, soft):
            return torch.cat((x, x, torch.zeros_like(x)), -1), {}

        def continuation(self, x, m):
            return x.new_ones(x.shape[0])

        def timeout(self, m):
            return torch.zeros(m.shape[0], dtype=torch.bool)

        def advance(self, x, m):
            return m + 1

    class WM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.7))

        def forward(self, x, u, c):
            return self.weight * x + 0.2 * u, torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1, 3)

    actor = torch.nn.Linear(1, 1, bias=False)
    critic = torch.nn.Linear(1, 2, bias=False)
    wm = WM()
    with torch.no_grad():
        actor.weight.fill_(0.3)
        critic.weight.fill_(0.5)
    start = {
        "x": torch.ones(1, 1),
        "memory": torch.zeros(1, 1),
        "context": torch.zeros(1, 1),
        "initial_contact": torch.zeros(1, 1, 1, 1),
        "initial_points": torch.zeros(1, 1, 1, 1, 3),
    }
    with freeze_parameters(wm, critic):
        loss, _ = imagine(Task(), wm, actor, critic, start, 3, 0.99, 0.5)
        loss.backward()
        grad = actor.weight.grad.item()

        def value(offset):
            with torch.no_grad():
                actor.weight.fill_(0.3 + offset)
            return imagine(Task(), wm, actor, critic, start, 3, 0.99, 0.5)[0].item()

        numerical = (value(1e-3) - value(-1e-3)) / 2e-3
    assert abs(grad - numerical) < 5e-4
    assert wm.weight.grad is None and critic.weight.grad is None
    assert wm.weight.requires_grad and critic.weight.requires_grad


def test_twin_lambda_timeout_bootstrap_and_failure():
    rewards = torch.ones(2, 2, 2)
    values = torch.full((3, 2, 2), 10.0)
    continuation = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    timeout = torch.tensor([[True, False], [False, False]])
    active = torch.tensor([[True, True], [False, True]])
    nxt = torch.full((2, 2, 2), 3.0)
    returns = lambda_returns(
        rewards, values, continuation, timeout, active, 0.9, 0.95, next_values=nxt
    )
    torch.testing.assert_close(returns[0, 0], torch.full((2,), 3.7))
    torch.testing.assert_close(returns[0, 1], torch.ones(2))


def test_replay_terminal_boundary_ring_and_resume():
    record = {
        "x": torch.zeros(3, 1),
        "episode": torch.zeros(3, dtype=torch.long),
        "done": torch.zeros(3, dtype=torch.bool),
        "terminated": torch.zeros(3, dtype=torch.bool),
        "valid": torch.ones(3, dtype=torch.bool),
    }
    replay = ReplayBuffer(record, 3, 10000, max_steps=4, minimum_steps=4)
    for t in range(7):
        record["x"].fill_(t)
        record["episode"].fill_(0 if t < 5 else 1)
        record["done"].fill_(t == 4)
        record["terminated"].fill_(t == 4)
        replay.add(record)
    sampled = replay.sample(256, 3, failure_fraction=0.25)
    for b in range(256):
        valid = sampled["mask"][b]
        assert sampled["episode"][b, valid].unique().numel() == 1
        done = sampled["done"][b].nonzero().flatten()
        if done.numel():
            assert not valid[done[0] + 1 :].any()
    restored = ReplayBuffer(record, 3, 10000, max_steps=4, minimum_steps=4)
    restored.load_state_dict(replay.state_dict())
    torch.testing.assert_close(restored.data["x"][restored.rows()], replay.data["x"][replay.rows()])


def test_dual_only_updates_on_real_16_step_clock():
    dual = DualController(2, "cpu")
    for _ in range(15):
        dual.observe(torch.ones(2), torch.ones(2, dtype=torch.bool))
    assert dual.updates == 0 and dual.weight == 0.5
    dual.observe(torch.ones(2), torch.ones(2, dtype=torch.bool))
    assert dual.updates == 1 and dual.weight < 0.5
    other = DualController(2, "cpu")
    other.load_state_dict(dual.state_dict())
    assert other.weight == dual.weight and list(other.returns) == list(dual.returns)


def test_capture_precedes_autoreset_and_reward_runs_once():
    state = torch.tensor([[1.0], [2.0]])
    count = [0]

    def compute(dt):
        count[0] += 1
        return torch.ones(2)

    env = SimpleNamespace(
        num_envs=2,
        reward_manager=SimpleNamespace(compute=compute, _step_reward=torch.ones(2, 3)),
        reset_buf=torch.tensor([True, False]),
        reset_terminated=torch.tensor([True, False]),
        reset_time_outs=torch.zeros(2, dtype=torch.bool),
    )
    adapter = IsaacAdapter.__new__(IsaacAdapter)
    adapter.env, adapter.device = env, "cpu"
    adapter.state = lambda: state.clone()
    adapter.memory = lambda: torch.zeros(2, 2)
    adapter.contacts = lambda: (torch.ones(2, 1, 1, 1), torch.zeros(2, 1, 1, 1, 3))
    adapter.context = torch.zeros(2, 1)
    adapter.episode = torch.zeros(2, dtype=torch.long)
    adapter.action_term = SimpleNamespace(
        processed_actions=torch.zeros(2, 1), raw_actions=torch.zeros(2, 1)
    )
    adapter.groups = {g: torch.tensor([i]) for i, g in enumerate(("rg", "rs", "rp"))}
    adapter.pending = adapter.captured = adapter._original_compute = None
    adapter.attach()
    adapter.begin()
    state.add_(10)
    env.reward_manager.compute(dt=1 / 60)
    state[0] = 0  # mimic upstream auto-reset after reward capture
    captured = adapter.finish()
    torch.testing.assert_close(captured["x1"], torch.tensor([[11.0], [12.0]]))
    assert count[0] == 1 and adapter.episode.tolist() == [1, 0]
    adapter.close()
