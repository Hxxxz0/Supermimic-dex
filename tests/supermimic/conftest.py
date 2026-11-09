import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
torch.set_num_threads(4)

from contrack_supermimic.state import TaskDescriptor
from contrack_supermimic.task import ConTrackTask


@pytest.fixture
def task_factory():
    def make(hands=1, objects=1, size=19, length=20, extra=0):
        d = TaskDescriptor(
            "sharpa" if size == 29 else "xhand",
            (size,) * hands,
            tuple(f"j{i}" for i in range(hands * size + extra)),
            tuple(f"o{i}" for i in range(objects)),
            tuple(tuple(f"h{h}l{i}" for i in range(10)) for h in range(hands)),
            4,
        )
        refs = {
            "q": torch.randn(length, hands, size) * 0.1,
            "pos": torch.randn(length, objects, 3) * 0.01,
            "quat": torch.nn.functional.normalize(torch.randn(length, objects, 4), dim=-1),
            "contact": torch.rand(length, hands, objects, 10) > 0.5,
            "points": torch.randn(length, hands, objects, 10, 3) * 0.01,
        }
        weights = {
            "tracking_arm": 10.0,
            "tracking_finger": 10.0,
            "tracking_obj_pos": 500.0,
            "tracking_obj_rot": 500.0,
            "contact_reward": 100.0,
            "contact_distance_reward": 100.0,
        }
        for prefix in ("qvel", "qacc"):
            for name, value in {
                "arm": 1 if prefix == "qvel" else 10,
                "finger": 1 if prefix == "qvel" else 10,
                "obj_pos": 1 if prefix == "qvel" else 100,
                "obj_rot": 0.1 if prefix == "qvel" else 10,
            }.items():
                weights[f"{prefix}_penalty_{name}"] = -value
        groups = {
            "rg": ["tracking_obj_pos", "tracking_obj_rot"],
            "rs": ["tracking_arm", "tracking_finger", "contact_reward", "contact_distance_reward"],
            "rp": [n for n in weights if "penalty" in n],
        }
        task = ConTrackTask(
            d, refs, 1.0, torch.tensor(([0.01] * 7 + [0.25] * (size - 7)) * hands), weights, groups
        )
        return task

    return make


def physical_state(task, batch=4):
    d = task.d
    x = torch.randn(batch, d.state_dim) * 0.03
    obj = x[:, 2 * d.joints :].reshape(batch, d.objects, 13)
    obj[..., 3:7] = torch.nn.functional.normalize(torch.randn(batch, d.objects, 4), dim=-1)
    return x
