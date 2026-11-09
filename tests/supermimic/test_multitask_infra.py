"""Ensure incomplete or misleading training artifacts cannot pass the new suite."""

import importlib.util
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

FILE = Path(__file__).resolve().parents[2] / "scripts/supermimic/multitask_case.py"
SPEC = importlib.util.spec_from_file_location("multitask_case", FILE)
case = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(case)


def artifacts(root, algorithm="supermimic"):
    metrics = dict(
        vector_steps=128, real_steps=8192, policy_updates=5, parity_max=dict(observation=0.0)
    )
    (root / "metrics.jsonl").write_text(json.dumps(metrics) + "\n")
    if algorithm == "supermimic":
        (root / "checkpoint_final.pt").touch()
        (root / "policy.pt").touch()
    else:
        (root / "model_7.pt").touch()
    return metrics


def test_complete_actor_and_ppo_artifacts(tmp_path):
    for algorithm in ("supermimic", "ppo"):
        run = tmp_path / algorithm
        run.mkdir()
        artifacts(run, algorithm)
        checkpoint, metrics = case.validate_training(run, algorithm, 8, 64)
        assert checkpoint.is_file() and metrics["real_steps"] == 8192


@pytest.mark.parametrize(
    "change",
    [
        dict(vector_steps=127),
        dict(real_steps=8191),
        dict(policy_updates=0),
        dict(actor_loss=float("nan")),
        dict(parity_max={}),
    ],
)
def test_incomplete_or_invalid_bptt_rejected(tmp_path, change):
    metrics = artifacts(tmp_path) | change
    (tmp_path / "metrics.jsonl").write_text(json.dumps(metrics) + "\n")
    with pytest.raises(RuntimeError):
        case.validate_training(tmp_path, "supermimic", 8, 64)


def test_failure_marker_rejected_even_with_complete_steps(tmp_path):
    artifacts(tmp_path)
    (tmp_path / "failure.json").write_text("{}")
    with pytest.raises(RuntimeError, match="reported failure"):
        case.validate_training(tmp_path, "supermimic", 8, 64)


def test_early_ppo_checkpoint_rejected(tmp_path):
    artifacts(tmp_path, "ppo")
    (tmp_path / "model_7.pt").rename(tmp_path / "model_6.pt")
    with pytest.raises(RuntimeError, match="end-of-budget"):
        case.validate_training(tmp_path, "ppo", 8, 64)


def test_nonfinite_telemetry_keeps_explicit_locations_and_rejects_results():
    original = dict(latest=dict(rg=float("nan"), losses=[float("inf"), 1.0]))
    out = case.telemetry(original)
    assert out["latest"] == dict(rg=None, losses=[None, 1.0])
    assert out["nonfinite_metrics"] == [
        dict(path="latest.rg", value="nan"),
        dict(path="latest.losses[0]", value="inf"),
    ]
    json.dumps(out, allow_nan=False)
    with pytest.raises(RuntimeError, match="Non-finite"):
        case.require_finite(original)


def test_process_failure_receipt_survives_nonfinite_final_metrics(tmp_path):
    runner = case.Runner(tmp_path, tmp_path, {})
    runner.python = ["python3"]
    run = tmp_path / "run"
    run.mkdir()
    command = ["-c", "import pathlib; pathlib.Path('run/metrics.jsonl').write_text('{\"rg\": NaN}'); raise SystemExit(3)"]
    with patch.object(case.subprocess, "check_output", return_value="GPU"):
        with pytest.raises(RuntimeError, match="smoke failed with exit status 3"):
            runner.execute("smoke", command, run)
    assert json.loads((tmp_path / "smoke_process.json").read_text())["exit_code"] == 3
    state = json.loads((tmp_path / "state.json").read_text())
    assert not state["alive"] and state["latest"]["rg"] is None
    assert state["nonfinite_metrics"] == [dict(path="latest.rg", value="nan")]


def test_frozen_training_contract_detects_edits(tmp_path):
    source = tmp_path / "trainer.py"
    source.write_text("original")
    plan = dict(frozen_training_files={"trainer.py": case.digest(source)})
    with patch.object(case, "git", side_effect=["commit", ""]):
        case.check_source(tmp_path, plan, "commit")
    source.write_text("changed")
    with patch.object(case, "git", side_effect=["commit", ""]):
        with pytest.raises(RuntimeError, match="Training contract changed"):
            case.check_source(tmp_path, plan, "commit")


def cluster_module(monkeypatch):
    monkeypatch.syspath_prepend(str(FILE.parent))
    spec = importlib.util.spec_from_file_location(
        "multitask_cluster", FILE.with_name("multitask_cluster.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_submission_is_single_gpu_without_obsolete_overrides(tmp_path, monkeypatch):
    cluster = cluster_module(monkeypatch)
    (tmp_path / "requests").mkdir()
    monkeypatch.setattr(cluster, "git", lambda *args: "commit")
    monkeypatch.setattr(cluster, "check_source", lambda *args: None)
    requests = []

    def create(method, flag, body):
        assert method == "CreateJob" and flag == "--body"
        requests.append(json.loads(body))
        return dict(JobId=f"job{len(requests)}")

    monkeypatch.setattr(cluster, "api", create)
    plan = dict(
        resource="obstacle_avoidance_4090",
        resource_id="quota",
        tasks=[dict(task="dexterhand-Cylinder_00-fps_20-xarm7_xhand-3600_3700")],
    )
    cluster.submit(tmp_path, plan, None)
    cluster.submit(tmp_path, plan, None)
    assert len(requests) == 2  # Idempotent: two algorithms, never duplicate jobs.
    for request in requests:
        assert request["JobSpecs"][0]["ResourceConfig"]["GPU"] == "1"
        assert "AssignNodeSpec" not in request["JobSpecs"][0]
        assert "Settings" not in request
        assert "--driver" not in request["UserCommand"]
        assert "--deny_nodes" not in request["UserCommand"]


def test_cloud_success_requires_evaluation_artifacts(tmp_path, monkeypatch):
    cluster = cluster_module(monkeypatch)
    monkeypatch.setattr(
        cluster, "api", lambda *args: dict(Jobs=[dict(JobId="job", Status="Succeeded")])
    )
    receipt = [
        dict(
            task="dexterhand-Cylinder_00",
            algorithm="supermimic",
            job_id="job",
            case_root=str(tmp_path / "case"),
        )
    ]
    assert not cluster.summarize(tmp_path, receipt)  # The remaining registered cases are missing.
    report = json.loads((tmp_path / "status.json").read_text())
    assert report["evaluated"] == 0
    assert not report["rows"][0]["evaluated"]


def test_materialized_resource_symlinks_keep_checkout_clean(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / ".gitignore").write_text((FILE.parents[2] / ".gitignore").read_text())
    (source / "configs").mkdir()
    (source / case.PLAN_PATH.relative_to(case.SOURCE)).write_text("{}")
    (source / "assets").mkdir()
    (source / "assets/README.md").write_text("Tracked resource parent, as in the real repo")
    subprocess.run(["git", "init", "--quiet", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "Fixture",
        ],
        check=True,
    )
    canonical = tmp_path / "resources"
    for name in (
        "data",
        "assets/meshes",
        "assets/urdf",
        "assets/usd",
        "assets/meshes_simplify_collision",
        "assets/urdf_simplify_collision",
    ):
        p = canonical / name
        p.mkdir(parents=True)
        (p / "resource").write_text("fixture")
    root = tmp_path / "case"
    root.mkdir()
    revision = case.git(source, "rev-parse", "HEAD")
    case.setup_case(source, root / "repo", root, canonical, revision)
    assert case.git(root / "repo", "status", "--porcelain") == ""
