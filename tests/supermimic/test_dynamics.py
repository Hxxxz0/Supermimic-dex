import torch

from contrack_supermimic.model import WorldModel
from contrack_supermimic.state import StateCodec, quaternion_error, rotvec_quat
from conftest import physical_state


def test_so3_identity_has_finite_derivatives():
    v = torch.zeros(3, 3, requires_grad=True)
    q = rotvec_quat(v)
    q.sum().backward()
    assert torch.isfinite(v.grad).all()
    torch.testing.assert_close(v.grad, torch.full_like(v, 0.5))


def test_state_integrate_roundtrip_and_quaternion_sign(task_factory):
    task = task_factory(hands=2, objects=3, extra=4)
    codec = StateCodec(task.d)
    x = physical_state(task)
    delta = torch.randn(4, task.d.delta_dim) * 0.002
    nxt = codec.integrate(x, delta)
    torch.testing.assert_close(codec.difference(nxt, x), delta, atol=2e-7, rtol=2e-4)
    _, _, obj = codec.unpack(nxt)
    torch.testing.assert_close(obj[..., 3:7].norm(dim=-1), torch.ones(4, 3))
    torch.testing.assert_close(
        quaternion_error(obj[..., 3:7], -obj[..., 3:7]), torch.zeros(4, 3), atol=1e-6, rtol=0
    )


def test_world_model_recursive_gradient_and_real_only_statistics(task_factory):
    task = task_factory(size=29, extra=29)
    wm = WorldModel(task.d, (32, 32), amp=False)
    x = physical_state(task)
    u = torch.randn(4, task.d.actions, requires_grad=True)
    context = torch.randn(4, task.d.context_dim)
    # Zero initialized output layers deliberately predict the identity before fitting.
    with torch.no_grad():
        wm.net[-1].weight.normal_(0, 0.005)
    for _ in range(8):
        x, _, _ = wm(x, u, context)
    x.square().mean().backward()
    assert torch.isfinite(u.grad).all() and u.grad.abs().sum() > 0
    assert wm.inputs.count == 0 and wm.deltas.count == 0
    wm.observe(physical_state(task), u.detach(), context, physical_state(task))
    assert wm.inputs.count == 4 and wm.deltas.count == 4
