#!/usr/bin/env python3
"""One-shot or hourly monitor; no GPU/log polling between checks."""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def tail(path, lines=100):
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 65536))
        return "\n".join(f.read().decode(errors="replace").splitlines()[-lines:])


def check(run_dir, pid):
    run_dir = Path(run_dir)
    alive = False
    if pid:
        try:
            os.kill(pid, 0)
            stat = Path(f"/proc/{pid}/stat")
            alive = not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"
        except ProcessLookupError:
            pass
    lines = tail(run_dir / "metrics.jsonl", 2).splitlines()
    latest = {}
    for line in reversed(lines):
        try:
            latest = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    stdout = tail(run_dir / "stdout.log", 120)
    anomalies = [
        line
        for line in stdout.splitlines()
        if any(
            word in line.lower()
            for word in ("traceback", "out of memory", "nan", "segmentation fault", "parity failed")
        )
    ]
    try:
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        gpu = "unavailable"
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    config = manifest.get("config", {})
    budget = config.get("budget_updates", config.get("max_iterations", 5000)) * 16
    complete = latest.get("vector_steps", 0) >= budget
    final = run_dir / "checkpoint_final.pt"
    if manifest.get("algorithm") == "ppo":
        candidates = list(run_dir.glob("model_[0-9]*.pt"))
        final = max(candidates, key=lambda p: int(p.stem.split("_")[1])) if candidates else final
    remaining = latest.get("estimated_remaining_seconds")
    if remaining is None and latest.get("vector_steps", 0) > 0:
        remaining = latest.get("wall_seconds", 0) * (budget / latest["vector_steps"] - 1)
    evaluation = run_dir / "evaluation.json"
    out = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "pid": pid,
        "alive": alive,
        "training_complete": complete,
        "final_checkpoint": final.exists(),
        "gpu": gpu,
        "latest": latest,
        "anomalies": anomalies,
        "estimated_remaining_seconds": max(0, remaining) if remaining is not None else None,
        "metrics_age_seconds": max(0, time.time() - (run_dir / "metrics.jsonl").stat().st_mtime)
        if lines
        else None,
        "evaluation": json.loads(evaluation.read_text()) if evaluation.exists() else None,
    }
    failure = run_dir / "failure.json"
    if failure.exists():
        out["failure"] = json.loads(failure.read_text())
    print(json.dumps(out, ensure_ascii=False), flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=3600)
    args = parser.parse_args()
    if args.interval < 3600:
        parser.error("Monitoring interval must be at least one hour")
    while True:
        result = check(args.run_dir, args.pid)
        if not args.watch or result["training_complete"] or (args.pid and not result["alive"]):
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
