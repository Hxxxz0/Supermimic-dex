#!/usr/bin/env python3
"""Log native finite-state and action parity diagnostics without altering dynamics.

Run with the Isaac Python launcher, e.g. diagnose_native.py train --output PATH
-- --algorithm supermimic --smoke --data DATA --embodiment sharpa ...
"""

import argparse
import json
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def tensor_summary(tensor):
    import torch

    tensor = tensor.detach()
    finite = torch.isfinite(tensor)
    values = tensor[finite]
    bad = (~finite).nonzero()[:16].cpu().tolist()
    return dict(
        shape=list(tensor.shape),
        nonfinite=int((~finite).sum()),
        bad_indices=bad,
        finite_min=float(values.min()) if values.numel() else None,
        finite_max=float(values.max()) if values.numel() else None,
    )


def install_capture(output):
    import torch
    from contrack_supermimic.isaac_adapter import IsaacAdapter

    output.parent.mkdir(parents=True, exist_ok=True)

    def emit(value):
        with output.open("a") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")

    def native(adapter):
        env = adapter.env
        fields = dict(
            state=adapter.state(), memory=adapter.memory(), context=adapter.context,
            ref_qpos=env.ref_qpos, ref_object_pos=env.ref_object_pos,
            ref_object_quat=env.ref_object_quat, reset_lib_state=env.reset_lib_state,
        )
        for name in dict.fromkeys(group[0] for group in adapter.read_groups):
            data = env.scene[name].data
            for attr in ("joint_pos", "joint_vel", "joint_pos_limits", "joint_stiffness",
                         "joint_damping", "joint_effort_limits", "root_state_w"):
                fields[f"{name}.{attr}"] = getattr(data, attr)
        observations = getattr(env.observation_manager, "_obs_buffer", {})
        for name, value in observations.items():
            if torch.is_tensor(value):
                fields[f"observation.{name}"] = value
        return {key: tensor_summary(value) for key, value in fields.items()}

    original_init, original_finish = IsaacAdapter.__init__, IsaacAdapter.finish

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._diagnostic_steps = 0
        self._diagnostic_invalid_steps = 0
        emit(dict(stage="initialized", fields=native(self), descriptor=self.d.to_dict()))

    def finish(self):
        record = original_finish(self)
        self._diagnostic_steps += 1
        invalid = not bool(record["valid"].all())
        if self._diagnostic_steps <= 3 or (invalid and self._diagnostic_invalid_steps < 3):
            fields = {key: tensor_summary(value) for key, value in record.items()
                      if torch.is_tensor(value) and key not in ("points", "contact")}
            expected = self.task.control(record["action"], record["memory"])
            finite = torch.isfinite(expected).all(-1) & torch.isfinite(record["u"]).all(-1)
            gap = (expected[finite] - record["u"][finite]).abs()
            emit(dict(stage="real_step", step=self._diagnostic_steps, fields=fields,
                      valid_rows=int(record["valid"].sum()), total_rows=self.env.num_envs,
                      finite_action_rows=int(finite.sum()),
                      action_gap_max=float(gap.max()) if gap.numel() else None,
                      native=native(self)))
            if invalid:
                self._diagnostic_invalid_steps += 1
                ids = (~record["valid"]).nonzero().flatten()[:8]
                torch.save({key: value[ids].detach().cpu() for key, value in record.items()
                            if torch.is_tensor(value) and value.ndim > 0
                            and value.shape[0] == self.env.num_envs},
                           output.with_name(f"invalid_step_{self._diagnostic_steps}.pt"))
        return record

    IsaacAdapter.__init__ = initialize
    IsaacAdapter.finish = finish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    args, remainder = parser.parse_known_args()
    if remainder[:1] == ["--"]:
        remainder.pop(0)
    install_capture(args.output)
    script = Path(__file__).with_name(args.command + ".py")
    sys.argv = [str(script), *remainder]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
