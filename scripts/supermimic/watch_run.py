#!/usr/bin/env python3
"""Hourly checks followed by automatic final real-physics evaluation."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from monitor import check

REPO = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--interval", type=int, default=3600)
    parser.add_argument("--episodes", type=int, default=4096)
    args = parser.parse_args()
    if args.interval < 3600:
        parser.error("Use at least one hour between checks")
    pid_info = json.loads((args.run_dir / "pid.json").read_text())
    pid = pid_info["pid"]
    while True:
        result = check(args.run_dir, pid)
        with (args.run_dir / "monitor.jsonl").open("a") as f:
            f.write(json.dumps(result) + "\n")
        if not result["alive"]:
            break
        time.sleep(args.interval)
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    if not result["training_complete"] or result.get("failure"):
        (args.run_dir / "completion.json").write_text(
            json.dumps({"state": "failed", "check": result}, indent=2) + "\n"
        )
        return 1
    if manifest["algorithm"] == "supermimic":
        checkpoint = args.run_dir / "policy.pt"
    else:
        candidates = list(args.run_dir.glob("model_[0-9]*.pt"))
        checkpoint = max(candidates, key=lambda p: int(p.stem.split("_")[1]))
    command = pid_info["command"]
    device = command[command.index("--device") + 1]
    embodiment = "sharpa" if "rby1a_sharpa" in Path(manifest["reference_path"]).name else "xhand"
    cmd = [
        "bash",
        str(REPO / "scripts/supermimic/python.sh"),
        str(REPO / "scripts/supermimic/evaluate.py"),
        "--checkpoint",
        str(checkpoint),
        "--data",
        manifest["reference_path"],
        "--embodiment",
        embodiment,
        "--device",
        device,
        "--num-envs",
        "128",
        "--episodes",
        str(args.episodes),
        "--output",
        str(args.run_dir / "evaluation.json"),
    ]
    env = os.environ.copy()
    visible = manifest["runtime"].get("cuda_visible_devices")
    if visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible
    with (args.run_dir / "evaluation.log").open("w") as log:
        process = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    success = process.returncode == 0 and (args.run_dir / "evaluation.json").exists()
    (args.run_dir / "completion.json").write_text(
        json.dumps(
            {
                "state": "evaluated" if success else "evaluation_failed",
                "evaluation_exit_code": process.returncode,
                "checkpoint": str(checkpoint),
            },
            indent=2,
        )
        + "\n"
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
