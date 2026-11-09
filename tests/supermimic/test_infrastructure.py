import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from contrack_supermimic.checkpoint import load_checkpoint, restore_environment


def script(name):
    path = Path(__file__).resolve().parents[2] / f"scripts/supermimic/{name}.py"
    spec = importlib.util.spec_from_file_location(f"infra_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_monitor_rejects_interrupted_checkpoint_and_partial_log(tmp_path, monkeypatch):
    monitor = script("monitor")
    monkeypatch.setattr(monitor.subprocess, "check_output", lambda *a, **kw: "0, 0, 0")
    (tmp_path / "manifest.json").write_text(json.dumps({"config": {"budget_updates": 8}}))
    (tmp_path / "metrics.jsonl").write_text(json.dumps({"vector_steps": 64}) + '\n{"partial":')
    (tmp_path / "checkpoint_final.pt").touch()
    assert not monitor.check(tmp_path, None)["training_complete"]
    (tmp_path / "metrics.jsonl").write_text(json.dumps({"vector_steps": 128}) + "\n")
    assert monitor.check(tmp_path, None)["training_complete"]


def test_evaluation_preallocates_trials_independent_of_episode_length():
    evaluate = script("evaluate")
    for episodes, num_envs in [(4096, 256), (7, 3), (2, 8)]:
        quotas = evaluate.episode_quotas(episodes, num_envs, "cpu")
        assert int(quotas.sum()) == episodes
        assert int(quotas.max() - quotas.min()) <= 1
        # Environment zero fails rapidly; its accepted trial count still has a
        # fixed cap, while every other environment receives its allocated trials.
        accepted = torch.zeros_like(quotas)
        for step in range(1, int(quotas.max()) * num_envs + 1):
            done = step % torch.arange(1, num_envs + 1) == 0
            accepted += (done & (accepted < quotas)).long()
        torch.testing.assert_close(accepted, quotas)


def test_paper_joint_error_is_mae_and_rmse_remains_separate():
    evaluate = script("evaluate")
    rmse, finger, arm = evaluate.joint_errors(
        torch.tensor([[3.0, 0.0, 2.0]]), torch.zeros(1, 3), torch.tensor([0]), torch.tensor([1, 2])
    )
    torch.testing.assert_close(finger, torch.tensor([1.0]))
    torch.testing.assert_close(arm, torch.tensor([3.0]))
    torch.testing.assert_close(rmse, torch.tensor([2.0**0.5]))


def test_checkpoint_rejects_different_reference_before_loading_parameters(tmp_path):
    data = tmp_path / "clip.h5"
    data.write_bytes(b"one reference")
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"format_version": 1, "reference_sha256": "different"}, checkpoint)
    trainer = SimpleNamespace(
        device="cpu",
        adapter=SimpleNamespace(env=SimpleNamespace(cfg=SimpleNamespace(reference_path=data))),
    )
    with pytest.raises(ValueError, match="reference clip"):
        load_checkpoint(trainer, checkpoint)


def test_restore_recreates_step_only_terminal_buffers(task_factory):
    task = task_factory()
    n = 2
    x = torch.zeros(n, task.d.state_dim)
    x[:, 2 * task.d.joints + 3] = 1

    class Scene(dict):
        def update(self, dt):
            pass

    scene = Scene(
        robot=SimpleNamespace(write_joint_state_to_sim=lambda *a, **kw: None),
        objects=SimpleNamespace(write_object_state_to_sim=lambda *a, **kw: None),
    )
    scene.env_origins = torch.zeros(n, 3)
    env = SimpleNamespace(scene=scene)
    adapter = SimpleNamespace(
        env=env,
        device="cpu",
        task=task,
        read_groups=[("robot", list(range(task.d.joints)))],
        episode=torch.zeros(n, dtype=torch.long),
        context=torch.zeros(n, task.d.context_dim),
    )
    state = {
        "physical": x,
        "tensors": {"reset_buf": torch.tensor([True, False])},
        "scalars": {},
        "adapter_episode": torch.zeros(n, dtype=torch.long),
        "context": adapter.context,
    }
    restore_environment(adapter, state)
    torch.testing.assert_close(env.reset_buf, torch.tensor([True, False]))


def test_calibration_picks_fastest_eligible_preset(tmp_path):
    module = script("suite")
    suite = module.Suite.__new__(module.Suite)
    suite.root, suite.commit = tmp_path, "test"
    results = []
    for path in module.clips("xhand", True):
        for preset, cost, state_loss in [
            ("e1", 1.0, 2.0),
            ("e2", 2.0, 1.0),
            ("e3", 3.0, 1.0),
            ("reference", 4.0, 1.0),
        ]:
            results.append(
                {
                    "algorithm": "supermimic",
                    "preset": preset,
                    "data": str(path),
                    "evaluation": {"success_rate": 0.8, "progress": 0.9},
                    "metrics": {
                        "holdout/state_loss": state_loss,
                        "holdout/point_loss": 1.0,
                        "holdout/contact_f1": 0.8,
                        "wall_seconds": cost,
                    },
                }
            )
    assert suite.select(results) == "e2"


def test_report_pairs_only_equal_real_budget_and_evaluation_protocol(tmp_path):
    report = script("report")
    results = []
    for algorithm in ("ppo", "supermimic"):
        run = tmp_path / algorithm
        run.mkdir()
        (run / "manifest.json").write_text(
            json.dumps({"git_sha": "same", "reference_sha256": "clip", "num_envs": 8000})
        )
        evaluation = {key: 0.5 for key in report.METRICS}
        evaluation.update(
            seed=12345,
            episodes=4096,
            num_envs=256,
            thresholds={"position_m": 0.1, "rotation_rad": 1.0},
            episode_allocation="fixed_per_environment",
            metric_protocol="same",
        )
        results.append(
            {
                "commit": "same",
                "run": str(run),
                "algorithm": algorithm,
                "seed": 0,
                "data": "clip.h5",
                "preset": "e1",
                "metrics": {"real_steps": 640000000},
                "evaluation": evaluation,
                "training_process_wall_seconds": 7200 if algorithm == "ppo" else 3600,
            }
        )
    paired = report.matched_pairs(results)
    assert len(paired) == 1 and paired[0]["wall_ratio_ppo_over_supermimic"] == 2.0
    results[1]["metrics"]["real_steps"] -= 8000
    assert not report.matched_pairs(results)
    results[1]["metrics"]["real_steps"] += 8000
    results[1]["evaluation"]["episodes"] = 128
    with pytest.raises(ValueError, match="protocol mismatch"):
        report.matched_pairs(results)


def test_curve_counts_completed_ppo_updates_and_exact_supermimic_steps(tmp_path):
    curves = script("evaluate_curves")
    for name in ("model_0.pt", "model_200.pt", "policy_66.pt", "policy_3202.pt"):
        (tmp_path / name).touch()
    assert [steps for steps, _ in curves.snapshots(tmp_path, "ppo")] == [16, 3216]
    assert [steps for steps, _ in curves.snapshots(tmp_path, "supermimic")] == [66, 3202]
