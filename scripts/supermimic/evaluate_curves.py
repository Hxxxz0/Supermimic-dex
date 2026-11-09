#!/usr/bin/env python3
"""Evaluate retained policy snapshots at coarse, matched real-budget milestones."""

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def snapshots(run, algorithm):
    pattern = "policy_[0-9]*.pt" if algorithm == "supermimic" else "model_[0-9]*.pt"
    result = []
    for path in run.glob(pattern):
        index = int(path.stem.split("_")[1])
        # Pinned upstream saves after completing update `index` (zero based).
        steps = index if algorithm == "supermimic" else (index + 1) * 16
        result.append((steps, path))
    return sorted(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--episodes", type=int, default=4096)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
    args = parser.parse_args()
    if any(not 0 < f < 1 for f in args.fractions):
        parser.error("Milestone fractions must be in (0,1)")
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    if manifest.get("resume_from"):
        parser.error("Resumed runs need a continuous wall-time lineage before curve evaluation")
    pid_info = json.loads((args.run_dir / "pid.json").read_text())
    if "started_utc" not in pid_info:
        parser.error("pid.json needs the training process started_utc timestamp")
    started = datetime.fromisoformat(pid_info["started_utc"]).timestamp()
    config = manifest["config"]
    budget = 16 * config.get("budget_updates", config.get("max_iterations", 5000))
    candidates = snapshots(args.run_dir, manifest["algorithm"])
    if not candidates:
        parser.error("No policy snapshots")
    selected = {min(candidates, key=lambda p: abs(p[0] - budget * f)) for f in args.fractions}
    env = os.environ.copy()
    visible = manifest["runtime"].get("cuda_visible_devices")
    if visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible
    embodiment = "sharpa" if "rby1a_sharpa" in Path(manifest["reference_path"]).name else "xhand"
    out = args.run_dir / "curves"
    out.mkdir(exist_ok=True)
    rows = []
    for steps, checkpoint in sorted(selected):
        output = out / f"evaluation_{steps}.json"
        if not output.exists():
            command = [
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
                args.device,
                "--num-envs",
                str(args.num_envs),
                "--episodes",
                str(args.episodes),
                "--output",
                str(output),
            ]
            with (out / f"evaluation_{steps}.log").open("w") as log:
                code = subprocess.run(
                    command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT
                ).returncode
            if code or not output.exists():
                raise RuntimeError(f"Snapshot evaluation failed: {checkpoint}")
        evaluation = json.loads(output.read_text())
        if evaluation["episodes"] != args.episodes or evaluation["num_envs"] != args.num_envs:
            raise ValueError("Cached curve uses different evaluation sizes")
        if evaluation["checkpoint"] != str(checkpoint.resolve()):
            raise ValueError("Cached curve uses a different checkpoint")
        wall = checkpoint.stat().st_mtime - started
        if wall < 0:
            raise ValueError("Checkpoint predates training process; cannot assign its wall time")
        rows.append(
            {
                "vector_steps": steps,
                "real_steps": steps * manifest["num_envs"],
                "training_process_wall_seconds": wall,
                "evaluation": evaluation,
            }
        )
    final = json.loads((args.run_dir / "evaluation.json").read_text())
    if final["episodes"] != args.episodes or final["num_envs"] != args.num_envs:
        raise ValueError("Final and checkpoint curves use different evaluation sizes")
    for row in rows:
        for field in (
            "reference_sha256",
            "seed",
            "thresholds",
            "episode_allocation",
            "metric_protocol",
        ):
            if row["evaluation"][field] != final[field]:
                raise ValueError(f"Final and checkpoint evaluation protocol differs: {field}")
    completion = json.loads((args.run_dir / "training_complete.json").read_text())
    rows.append(
        {
            "vector_steps": budget,
            "real_steps": budget * manifest["num_envs"],
            "training_process_wall_seconds": completion["training_process_wall_seconds"],
            "evaluation": final,
        }
    )
    (args.run_dir / "curves.json").write_text(
        json.dumps(
            {
                "milestone_fractions": args.fractions,
                "time_to_quality_semantics": "first observed passing checkpoint; earlier unsampled crossings are unknown",
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"run": str(args.run_dir), "evaluated_snapshots": len(rows)}))


if __name__ == "__main__":
    main()
