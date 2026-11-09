#!/usr/bin/env python3
"""Verify successful video selections against independent BPTT evaluations."""

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(repo=REPO):
    report = json.loads((repo / "results/xhand_screen_20261109/results.json").read_text())
    for name in ("wineglass", "waterbottle", "waffleiron", "cubemedium", "sharpa_cubemedium"):
        sharpa = name.startswith("sharpa_")
        root = repo / "results" / f"{name}_e3"
        evaluation = json.loads((root / "evaluation.json").read_text())
        manifest = json.loads((root / "manifest.json").read_text())
        receipt = json.loads((repo / "docs/media" / f"{name}_bptt_success.json").read_text())
        video = repo / "docs/media" / f"{name}_bptt_success.mp4"
        assert receipt["algorithm"] == manifest["algorithm"] == "supermimic"
        assert receipt["embodiment"] == ("sharpa" if sharpa else "xhand")
        assert (
            receipt["checkpoint_sha256"]
            == evaluation["checkpoint_sha256"]
            == digest(root / "policy.pt")
        )
        assert receipt["reference_sha256"] == evaluation["reference_sha256"]
        assert receipt["video_sha256"] == digest(video)
        assert (
            receipt["thresholds"]
            == evaluation["thresholds"]
            == {"position_m": 0.1, "rotation_rad": 1.0}
        )
        assert evaluation["episodes"] == 4096 and receipt["evaluation_seed"] == 12345
        assert receipt["selection"] == "first successful episodes; all attempts recorded"
        assert len(receipt["episodes"]) == 3 and all(row["success"] for row in receipt["episodes"])
        assert [row["attempt"] for row in receipt["attempts"] if row["success"]] == [
            row["attempt"] for row in receipt["episodes"]
        ]
        matching_report = report
        if sharpa:
            matching_report = json.loads(
                (repo / "results/sharpa_screen_20261109/results.json").read_text()
            )
        matches = [
            row
            for row in matching_report["cases"]
            if row["algorithm"] == "supermimic"
            and row["evaluation"] is not None
            and row["evaluation"]["reference_sha256"] == receipt["reference_sha256"]
        ]
        assert (
            len(matches) == 1
            and matches[0]["evaluation"]["success_rate"] == evaluation["success_rate"]
        )
        assert (
            not manifest["smoke"] and manifest["resume_from"] is None and not manifest["git_status"]
        )
        assert manifest["seed"] == 2026 and manifest["num_envs"] == (6000 if sharpa else 8000)
        assert receipt["frames"] > 0 and receipt["duration_seconds"] > 0
        if receipt["prescan_steps"]:
            streak = 0
            for outcome in receipt["prescan_outcomes"]:
                if outcome["environment_index"] == receipt["environment_index"]:
                    streak = streak + 1 if outcome["success"] else 0
            assert streak >= receipt.get("prescan_required_successes", 1)
    print(
        json.dumps(
            dict(
                verified=True,
                videos=5,
                method="BPTT",
                selection="Representative successful episodes; independent full rates retained",
            )
        )
    )


if __name__ == "__main__":
    verify()
