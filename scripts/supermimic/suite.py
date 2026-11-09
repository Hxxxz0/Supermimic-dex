#!/usr/bin/env python3
"""Event-driven four-GPU suite, with hourly health checks and matched real budgets."""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from contrack_supermimic.experiment import RESULT_ROOT, git


PILOTS = ("dexterhand-Cuboid_00", "grab-s1_hammer_use_1", "arctic-s01_box_use_01")
# Published Table 3 progress, used as a baseline reproduction diagnostic.
PILOT_PROGRESS = dict(zip(PILOTS, (0.745, 0.984, 1.0)))


def clips(embodiment, pilot=False):
    found = sorted((REPO / "data" / embodiment).glob("*.h5"))
    if len(found) != 13:
        raise RuntimeError(f"Expected 13 {embodiment} clips; found {len(found)}")
    return (
        [next(p for p in found if p.name.startswith(prefix)) for prefix in PILOTS]
        if pilot
        else found
    )


def last_metrics(run):
    path = run / "metrics.jsonl"
    return json.loads(path.read_text().splitlines()[-1]) if path.exists() else {}


def gpu_inventory():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return {
        int(i): (int(total) - int(used), int(util))
        for i, total, used, util in (
            line.replace(" ", "").split(",") for line in output.splitlines()
        )
    }


class Suite:
    def __init__(self, args):
        self.args = args
        self.commit = git(REPO, "rev-parse", "HEAD")
        if git(REPO, "status", "--porcelain"):
            raise RuntimeError("Commit source before scheduling a suite")
        self.root = args.output or RESULT_ROOT / "runs" / f"suite-{self.commit[:10]}"
        self.root.mkdir(parents=True, exist_ok=True)
        identity = self.root / "suite_manifest.json"
        if identity.exists():
            if json.loads(identity.read_text())["commit"] != self.commit:
                raise RuntimeError("Suite directory belongs to a different source revision")
        else:
            identity.write_text(
                json.dumps(
                    {
                        "commit": self.commit,
                        "created_utc": datetime.now(timezone.utc).isoformat(),
                        "calibration_seed": 2026,
                        "evaluation_seed": 12345,
                        "seeds": [0, 1, 2],
                        "num_envs": {"xhand": 8000, "sharpa": 6000},
                        "budget_updates": {"calibration": 500, "pilot": 5000, "full": 5000},
                        "evaluation_episodes": 4096,
                    },
                    indent=2,
                )
                + "\n"
            )
        self.results = []
        self.next_check = time.monotonic()

    def state(self, stage, pending, active):
        data = {
            "checked_utc": datetime.now(timezone.utc).isoformat(),
            "commit": self.commit,
            "stage": stage,
            "pending": len(pending),
            "active": [{"gpu": gpu, "run": str(job["run"])} for job, gpu in active.values()],
            "results": self.results,
            "gpu_inventory": gpu_inventory(),
            "required_free_mib": self.args.min_free_mib,
        }
        tmp = self.root / "state.json.tmp"
        tmp.write_text(json.dumps(data, indent=2) + "\n")
        os.replace(tmp, self.root / "state.json")
        print(json.dumps(data), flush=True)

    def jobs(self, stage, paths, seeds, presets, budget):
        out = []
        for path in paths:
            embodiment = path.parent.name
            for seed in seeds:
                for algorithm, preset in [("ppo", "e1")] + [("supermimic", p) for p in presets]:
                    run = self.root / stage / path.stem / f"{algorithm}-{preset}-seed{seed}"
                    out.append(
                        {
                            "stage": stage,
                            "data": str(path),
                            "embodiment": embodiment,
                            "algorithm": algorithm,
                            "preset": preset,
                            "seed": seed,
                            "budget": budget,
                            "run": run,
                        }
                    )
        return out

    def execute(self, job, gpu):
        run = job["run"]
        run.mkdir(parents=True, exist_ok=True)
        completion = run / "completion.json"
        if completion.exists():
            result = json.loads(completion.read_text())
            if result.get("state") == "evaluated":
                return result
            raise RuntimeError(f"Failed previous run needs diagnosis: {run}")
        if (run / "manifest.json").exists():
            raise RuntimeError(f"Interrupted previous run needs diagnosis/resume: {run}")
        if git(REPO, "rev-parse", "HEAD") != self.commit or git(REPO, "status", "--porcelain"):
            raise RuntimeError("Source changed during suite; restart under a new revision")
        num_envs = 8000 if job["embodiment"] == "xhand" else 6000
        cmd = [
            "bash",
            str(REPO / "scripts/supermimic/python.sh"),
            str(REPO / "scripts/supermimic/train.py"),
            "--algorithm",
            job["algorithm"],
            "--preset",
            job["preset"],
            "--embodiment",
            job["embodiment"],
            "--data",
            job["data"],
            "--seed",
            str(job["seed"]),
            "--budget-updates",
            str(job["budget"]),
            "--num-envs",
            str(num_envs),
            "--device",
            "cuda:0",
            "--run-dir",
            str(run),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        started = time.perf_counter()
        started_utc = datetime.now(timezone.utc).isoformat()
        with (run / "stdout.log").open("w") as log:
            process = subprocess.Popen(
                cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            (run / "pid.json").write_text(
                json.dumps(
                    {"pid": process.pid, "command": cmd, "started_utc": started_utc}, indent=2
                )
                + "\n"
            )
            code = process.wait()  # OS wait, not polling
        training_wall = time.perf_counter() - started
        metrics = last_metrics(run)
        if (
            code
            or (run / "failure.json").exists()
            or metrics.get("vector_steps", 0) < job["budget"] * 16
        ):
            result = {"state": "failed", "run": str(run), "exit_code": code, "metrics": metrics}
            completion.write_text(json.dumps(result, indent=2) + "\n")
            raise RuntimeError(f"Training failed/incomplete: {run}")
        if job["algorithm"] == "supermimic" and not (run / "checkpoint_final.pt").is_file():
            raise RuntimeError(f"Missing final dynamics checkpoint: {run}")
        (run / "training_complete.json").write_text(
            json.dumps(
                {
                    "training_process_wall_seconds": training_wall,
                    "metrics": metrics,
                    "commit": self.commit,
                },
                indent=2,
            )
            + "\n"
        )
        checkpoint = (
            run / "policy.pt"
            if job["algorithm"] == "supermimic"
            else max(run.glob("model_[0-9]*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        )
        eval_cmd = [
            "bash",
            str(REPO / "scripts/supermimic/python.sh"),
            str(REPO / "scripts/supermimic/evaluate.py"),
            "--checkpoint",
            str(checkpoint),
            "--data",
            job["data"],
            "--embodiment",
            job["embodiment"],
            "--device",
            "cuda:0",
            "--num-envs",
            "256",
            "--episodes",
            "4096",
            "--output",
            str(run / "evaluation.json"),
        ]
        with (run / "evaluation.log").open("w") as log:
            code = subprocess.run(
                eval_cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT
            ).returncode
        if code or not (run / "evaluation.json").exists():
            raise RuntimeError(f"Evaluation failed: {run}")
        if job["algorithm"] == "supermimic":
            wm_cmd = [
                "bash",
                str(REPO / "scripts/supermimic/python.sh"),
                str(REPO / "scripts/supermimic/evaluate_wm.py"),
                str(run / "checkpoint_final.pt"),
                "--device",
                "cuda:0",
                "--output",
                str(run / "wm_evaluation.json"),
            ]
            with (run / "wm_evaluation.log").open("w") as log:
                code = subprocess.run(
                    wm_cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT
                ).returncode
            if code:
                raise RuntimeError(f"WM evaluation failed: {run}")
        if job["stage"] == "pilot":
            curve_cmd = [
                "bash",
                str(REPO / "scripts/supermimic/python.sh"),
                str(REPO / "scripts/supermimic/evaluate_curves.py"),
                str(run),
                "--device",
                "cuda:0",
            ]
            with (run / "curves.log").open("w") as log:
                code = subprocess.run(
                    curve_cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT
                ).returncode
            if code:
                raise RuntimeError(f"Checkpoint curve evaluation failed: {run}")
        result = {
            "state": "evaluated",
            "commit": self.commit,
            "run": str(run),
            "algorithm": job["algorithm"],
            "preset": job["preset"],
            "seed": job["seed"],
            "data": job["data"],
            "metrics": metrics,
            "evaluation": json.loads((run / "evaluation.json").read_text()),
            "training_process_wall_seconds": training_wall,
            "training_gpu_hours": training_wall / 3600,
            "total_process_wall_seconds": time.perf_counter() - started,
        }
        completion.write_text(json.dumps(result, indent=2) + "\n")
        return result

    def dispatch(self, stage, jobs):
        pending = []
        active = {}
        completed = []
        for job in jobs:
            completion = job["run"] / "completion.json"
            if (
                completion.exists()
                and json.loads(completion.read_text()).get("state") == "evaluated"
            ):
                result = json.loads(completion.read_text())
                if result.get("commit") != self.commit:
                    raise RuntimeError(
                        f"Cached result belongs to a different source revision: {completion}"
                    )
                completed.append(result)
                self.results.append(result)
            else:
                pending.append(job)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.args.gpus)) as pool:
            while pending or active:
                due = time.monotonic() >= self.next_check
                # Check GPU availability only at hourly checks or immediately after an
                # owned job completion; all other waiting uses OS/future notifications.
                inventory = gpu_inventory()
                used = {gpu for _, gpu in active.values()}
                for gpu in self.args.gpus:
                    free, util = inventory.get(gpu, (0, 100))
                    if (
                        pending
                        and gpu not in used
                        and free >= self.args.min_free_mib
                        and util <= 10
                    ):
                        job = pending.pop(0)
                        active[pool.submit(self.execute, job, gpu)] = (job, gpu)
                if due:
                    from monitor import check

                    for job, gpu in active.values():
                        pid_path = job["run"] / "pid.json"
                        if pid_path.exists():
                            status = check(job["run"], json.loads(pid_path.read_text())["pid"])
                            with (self.root / "monitor.jsonl").open("a") as f:
                                f.write(json.dumps(status) + "\n")
                    self.state(stage, pending, active)
                    self.next_check = time.monotonic() + 3600
                timeout = max(0.01, self.next_check - time.monotonic())
                if active:
                    done, _ = concurrent.futures.wait(
                        active, timeout=timeout, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for future in done:
                        job, gpu = active.pop(future)
                        try:
                            result = future.result()
                        except BaseException as error:
                            (self.root / "failure.json").write_text(
                                json.dumps(
                                    {
                                        "error": str(error),
                                        "run": str(job["run"]),
                                        "commit": self.commit,
                                    },
                                    indent=2,
                                )
                                + "\n"
                            )
                            raise
                        completed.append(result)
                        self.results.append(result)
                        self.state(stage, pending, active)
                else:
                    time.sleep(timeout)
        return completed

    def select(self, calibration):
        candidates = []
        for preset in ("e1", "e2", "e3", "reference"):
            eligible = True
            cost = 0.0
            for path in clips("xhand", True):
                same = [r for r in calibration if r["data"] == str(path)]
                reference = next(
                    r for r in same if r["algorithm"] == "supermimic" and r["preset"] == "reference"
                )
                candidate = next(
                    r for r in same if r["algorithm"] == "supermimic" and r["preset"] == preset
                )
                a, b = candidate["evaluation"], reference["evaluation"]
                ma, mb = candidate["metrics"], reference["metrics"]
                eligible &= (
                    a["success_rate"] >= b["success_rate"] - 0.05
                    and a["progress"] >= b["progress"] - 0.05
                )
                eligible &= ma["holdout/state_loss"] <= 1.21 * mb["holdout/state_loss"]
                eligible &= ma["holdout/point_loss"] <= 1.21 * mb["holdout/point_loss"]
                eligible &= ma["holdout/contact_f1"] >= mb["holdout/contact_f1"] - 0.02
                cost += ma["wall_seconds"]
            candidates.append(
                {"preset": preset, "eligible": bool(eligible), "summed_training_wall_seconds": cost}
            )
        choice = min(
            (x for x in candidates if x["eligible"]),
            key=lambda x: x["summed_training_wall_seconds"],
        )["preset"]
        selected = {"preset": choice, "commit": self.commit, "candidates": candidates}
        (self.root / "selected_preset.json").write_text(json.dumps(selected, indent=2) + "\n")
        return choice

    def run(self):
        cal_jobs = self.jobs(
            "calibration", clips("xhand", True), [2026], ["e1", "e2", "e3", "reference"], 500
        )
        calibration = self.dispatch("calibration", cal_jobs)
        preset = self.select(calibration)
        if self.args.stage == "calibrate":
            return
        pilot = self.dispatch(
            "pilot", self.jobs("pilot", clips("xhand", True), [0, 1, 2], [preset], 5000)
        )
        gaps = []
        for path in clips("xhand", True):
            ppo = [
                r["evaluation"] for r in pilot if r["data"] == str(path) and r["algorithm"] == "ppo"
            ]
            sm = [
                r["evaluation"]
                for r in pilot
                if r["data"] == str(path) and r["algorithm"] == "supermimic"
            ]

            def mean(values, key):
                return sum(v[key] for v in values) / len(values)

            expected = next(v for k, v in PILOT_PROGRESS.items() if path.name.startswith(k))
            gaps.append(
                {
                    "clip": path.name,
                    "success_gap": mean(sm, "success_rate") - mean(ppo, "success_rate"),
                    "progress_gap": mean(sm, "progress") - mean(ppo, "progress"),
                    "ppo_progress": mean(ppo, "progress"),
                    "paper_progress": expected,
                    "baseline_reproduction_diagnostic": mean(ppo, "progress") >= expected - 0.15,
                }
            )
        gate = {
            "passed": all(
                x["success_gap"] >= -0.05
                and x["progress_gap"] >= -0.05
                and x["baseline_reproduction_diagnostic"]
                for x in gaps
            ),
            "gaps": gaps,
        }
        (self.root / "pilot_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
        if not gate["passed"]:
            raise RuntimeError("Pilot quality gate failed; diagnose before expanding")
        if self.args.stage == "pilot":
            return
        remaining = [p for p in clips("xhand") if p not in clips("xhand", True)] + clips("sharpa")
        self.dispatch("full", self.jobs("full", remaining, [0, 1, 2], [preset], 5000))
        (self.root / "completed.json").write_text(
            json.dumps({"commit": self.commit, "results": self.results}, indent=2) + "\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("calibrate", "pilot", "all"), default="pilot")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--min-free-mib", type=int, default=29000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.gpus or len(set(args.gpus)) != len(args.gpus):
        parser.error("Provide distinct GPU indices")
    if args.min_free_mib <= 0:
        parser.error("--min-free-mib must be positive")
    suite = Suite(args)
    try:
        suite.run()
    except BaseException as error:
        (suite.root / "failure.json").write_text(
            json.dumps({"error": str(error), "commit": suite.commit}, indent=2) + "\n"
        )
        raise


if __name__ == "__main__":
    main()
