import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from contrack_supermimic.state import quat_conjugate, quat_rotate_inverse
from conftest import physical_state


@pytest.mark.parametrize("size,hands,objects", [(19, 1, 1), (19, 2, 3), (29, 1, 2), (29, 2, 2)])
def test_every_reward_against_pinned_upstream(task_factory, monkeypatch, size, hands, objects):
    torch.manual_seed(17)
    task = task_factory(hands, objects, size)
    x = physical_state(task)
    q, v, obj = task.codec.unpack(x)
    memory = task.memory(
        torch.arange(4).float(),
        torch.randn(4, task.d.actions + 6 * objects) * 0.01,
        torch.tensor([[0.1, 1.0]]).repeat(4, 1),
    )
    contact = (torch.rand(4, hands, objects, 10) > 0.5).float()
    points = torch.randn(4, hands, objects, 10, 3) * 0.01
    _, terms = task.rewards(x, memory, contact, points, soft=False)
    monkeypatch.setitem(
        sys.modules, "isaaclab.utils.math", SimpleNamespace(quat_apply_inverse=quat_rotate_inverse)
    )
    repo = Path(__file__).resolve().parents[2]
    embodiment = "mimic_xarm_xhand" if size == 19 else "mimic_rby1a_sharpa"
    path = repo / f"source/ConTrack/ConTrack/tasks/manager_based/{embodiment}/mdp/rewards.py"
    spec = importlib.util.spec_from_file_location("upstream_reward_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scene = {
        f"h{h}": SimpleNamespace(
            data=SimpleNamespace(
                joint_pos=q[:, h * size : (h + 1) * size], joint_vel=v[:, h * size : (h + 1) * size]
            )
        )
        for h in range(hands)
    }
    origins = torch.zeros(4, 3)
    scene["objects"] = SimpleNamespace(
        data=SimpleNamespace(
            object_pos_w=obj[..., :3],
            object_quat_w=obj[..., 3:7],
            object_lin_vel_w=obj[..., 7:10],
            object_ang_vel_w=obj[..., 10:13],
        )
    )
    for h, keys in enumerate(task.d.contact_links):
        for link, key in enumerate(keys):
            quat = obj[..., 3:7]
            world = quat_rotate_inverse(quat_conjugate(quat), points[:, h, :, link]) + obj[..., :3]
            world = torch.where(contact[:, h, :, link, None].bool(), world, float("nan"))
            scene[key] = SimpleNamespace(data=SimpleNamespace(contact_pos_w=world[:, None]))

    class Scene(dict):
        pass

    scene = Scene(scene)
    scene.env_origins = origins
    prev = memory[:, 1:-2]
    env = SimpleNamespace(
        scene=scene,
        num_envs=4,
        device="cpu",
        frame_idx=memory[:, 0].long(),
        ref_len=task.length,
        ref_qpos=task.ref["q"],
        ref_object_pos=task.ref["pos"],
        ref_object_quat=task.ref["quat"],
        ref_hands=[(f"h{h}", list(range(size)), h) for h in range(hands)],
        prev_arm_qvel=prev[:, task.arm],
        prev_finger_qvel=prev[:, task.finger],
        prev_object_lin_vel=prev[:, task.d.actions :].reshape(4, objects, 6)[..., :3],
        prev_object_ang_vel=prev[:, task.d.actions :].reshape(4, objects, 6)[..., 3:],
        cfg=SimpleNamespace(sim=SimpleNamespace(dt=1 / 120), decimation=2),
    )
    if size == 19:
        env.xhand_contact_sensor_keys = {
            f"h{h}": list(keys) for h, keys in enumerate(task.d.contact_links)
        }
        env.ref_xhand_is_contact = task.ref["contact"].permute(0, 2, 1, 3).bool()
        env.ref_xhand_contact_points = task.ref["points"].permute(0, 2, 1, 3, 4)
    else:
        # Sharpa's sensor-key lookup uses is_rhand rather than asset name.
        env.is_rhand = torch.tensor([True, False][:hands])
        env.hand_contact_sensor_keys = {
            side: list(keys) for side, keys in zip(("right", "left"), task.d.contact_links)
        }
        env.ref_hand_is_contact = task.ref["contact"].permute(0, 2, 1, 3).bool()
        env.ref_hand_contact_points = task.ref["points"].permute(0, 2, 1, 3, 4)
    for name in task.weights:
        kwargs = (
            {"std": task.pos_std if name == "tracking_obj_pos" else task.rot_std}
            if name in ("tracking_obj_pos", "tracking_obj_rot")
            else {}
        )
        expected = getattr(module, name)(env, **kwargs) * task.weights[name] * task.d.dt
        torch.testing.assert_close(terms[name], expected, atol=2e-5, rtol=2e-5)


def test_soft_binary_limit_and_empty_contacts(task_factory):
    task = task_factory(hands=2, objects=2)
    x = physical_state(task)
    memory = task.memory(
        torch.zeros(4), task.previous_velocity(x), torch.tensor([[0.1, 1.0]]).repeat(4, 1)
    )
    for p in (torch.zeros(4, 2, 2, 10), (torch.rand(4, 2, 2, 10) > 0.5).float()):
        points = torch.randn(4, 2, 2, 10, 3)
        _, hard = task.rewards(x, memory, p, points, False)
        _, soft = task.rewards(x, memory, p, points, True)
        for name in ("contact_reward", "contact_distance_reward"):
            torch.testing.assert_close(hard[name], soft[name])
        assert torch.isfinite(torch.stack(list(soft.values()))).all()
    tiny = torch.full((4, 2, 2, 10), 1e-6, requires_grad=True)
    perfect = task.ref["points"][torch.zeros(4, dtype=torch.long)]
    _, soft = task.rewards(x, memory, tiny, perfect, True)
    assert soft["contact_distance_reward"].max() < 1e-4
    soft["contact_distance_reward"].sum().backward()
    assert torch.isfinite(tiny.grad).all()


def test_observation_order_future_clamp_and_phase(task_factory):
    task = task_factory(hands=2, objects=2)
    x = physical_state(task)
    memory = task.memory(
        torch.tensor([0.0, 1.5, 18.0, 19.0]), task.previous_velocity(x), torch.ones(4, 2)
    )
    p = torch.rand(4, 2, 2, 10, requires_grad=True)
    obs = task.observation(x, memory, p)
    assert obs.shape == (4, task.d.observation_dim)
    torch.testing.assert_close(obs[:, -1], torch.tensor([0.0, 1 / 19, 18 / 19, 1.0]))
    obs.sum().backward()
    assert p.grad.abs().sum() > 0


@pytest.mark.parametrize("embodiment", ["mimic_xarm_xhand", "mimic_rby1a_sharpa"])
def test_upstream_velocity_cache_behavior_is_preserved(task_factory, embodiment):
    import ast

    repo = Path(__file__).resolve().parents[2]
    path = repo / f"source/ConTrack/ConTrack/tasks/manager_based/{embodiment}/mdp/state.py"
    tree = ast.parse(path.read_text())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "advance_reference_frame"
    )
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), scope)
    size = 19 if embodiment == "mimic_xarm_xhand" else 29
    task = task_factory(size=size)
    state = physical_state(task, batch=4)
    q, v, obj = task.codec.unpack(state)

    class Scene(dict):
        pass

    scene = Scene(
        hand=SimpleNamespace(data=SimpleNamespace(joint_pos=q, joint_vel=v)),
        objects=SimpleNamespace(
            data=SimpleNamespace(
                object_pos_w=obj[..., :3],
                object_quat_w=obj[..., 3:7],
                object_lin_vel_w=obj[..., 7:10],
                object_ang_vel_w=obj[..., 10:13],
            )
        ),
    )
    scene.env_origins = torch.zeros(4, 3)
    env = SimpleNamespace(
        reset_buf=torch.zeros(4),
        scene=scene,
        ref_hands=[("hand", list(range(size)), 0)],
        frame_idx=torch.zeros(4, dtype=torch.long),
        episode_start_frame=torch.zeros(4, dtype=torch.long),
        episode_states=torch.zeros(4, 20, 2 * size + 13),
        episode_filled=torch.zeros(4, 20, dtype=torch.bool),
        prev_arm_qvel=torch.ones(4, 7),
        prev_finger_qvel=torch.ones(4, size - 7),
        prev_object_lin_vel=torch.ones(4, 1, 3),
        prev_object_ang_vel=torch.ones(4, 1, 3),
        frame_idx_f=torch.zeros(4),
        ref_step_frames=1.0,
        ref_len=20,
    )
    scope["advance_reference_frame"](env, None)
    assert (env.prev_arm_qvel == 1).all() and (env.prev_finger_qvel == 1).all()
    memory = task.memory(torch.zeros(4), torch.ones(4, size + 6), torch.ones(4, 2))
    next_memory = task.advance(state, memory)
    torch.testing.assert_close(next_memory[:, 1:-2], memory[:, 1:-2])
