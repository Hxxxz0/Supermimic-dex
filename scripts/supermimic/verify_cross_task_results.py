#!/usr/bin/env python3
"""Check published cross-task evaluation records and the two new BPTT policies."""

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(repo=REPO):
    repo = Path(repo)
    report = json.loads((repo / "results/xhand_screen_20261109/results.json").read_text())
    cases = report["cases"]
    assert len(cases) == 20
    assert len({(row["task"], row["algorithm"]) for row in cases}) == 20
    assert len({row["task"] for row in cases}) == 10
    for name, expected in report["frozen_training_files"].items():
        assert sha256(repo / name) == expected, name
    for row in cases:
        evaluation = row["evaluation"]
        assert evaluation["episodes"] == 4096 and evaluation["seed"] == 12345
        assert evaluation["episode_allocation"] == "fixed_per_environment"
        assert evaluation["thresholds"] == {"position_m": 0.1, "rotation_rad": 1.0}
        assert row["vector_steps"] == 8000 and row["real_steps"] == 64000000
        assert row["initialization_verified"] == "from_scratch"
        assert not row["git_status_at_training"]
        assert 0 <= evaluation["success_rate"] <= 1
        if row["algorithm"] == "supermimic":
            assert row["policy_updates"] == 418
    config = json.loads((repo / "configs/hammer_best.json").read_text())
    policies = {}
    for name, successes in (("wineglass_e3", 4095), ("waterbottle_e3", 4081)):
        root = repo / "results" / name
        manifest = json.loads((root / "manifest.json").read_text())
        evaluation = json.loads((root / "evaluation.json").read_text())
        assert manifest["algorithm"] == "supermimic"
        assert manifest["resume_from"] is None and not manifest["smoke"]
        assert manifest["seed"] == 2026 and manifest["num_envs"] == 8000
        assert not manifest["git_status"]
        assert all(manifest["config"][key] == value for key, value in config.items())
        assert evaluation["success_rate"] == successes / 4096
        assert evaluation["checkpoint_sha256"] == sha256(repo / evaluation["checkpoint"])
        assert manifest["reference_sha256"] == evaluation["reference_sha256"]
        rows = [
            row
            for row in cases
            if row["algorithm"] == "supermimic"
            and row["evaluation"]["reference_sha256"] == evaluation["reference_sha256"]
        ]
        assert len(rows) == 1 and rows[0]["evaluation"] == evaluation
        reference = repo / evaluation["reference"]
        if reference.is_file():
            assert sha256(reference) == evaluation["reference_sha256"]
        policies[name] = evaluation["checkpoint_sha256"]
    result = dict(
        verified=True,
        cases=20,
        frozen_files=len(report["frozen_training_files"]),
        policies=policies,
        scope="Artifact/source integrity; recorded single-seed results, no new training",
    )
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    verify()
