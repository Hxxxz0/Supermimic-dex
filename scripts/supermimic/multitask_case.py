#!/usr/bin/env python3
"""Run one frozen-recipe task/algorithm on its allocated DLC GPU."""

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
PLAN_PATH = SOURCE / "configs/sharpa_20261109.json"


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def check_source(repo, plan, expected_commit):
    if git(repo, "rev-parse", "HEAD") != expected_commit or git(repo, "status", "--porcelain"):
        raise RuntimeError("Expected the exact clean committed experiment source")
    for name, expected in plan["frozen_training_files"].items():
        if digest(repo / name) != expected:
            raise RuntimeError(f"Training contract changed: {name}")


def last_metrics(run):
    path = run / "metrics.jsonl"
    if not path.exists():
        return {}
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 65536))
        lines = stream.read().decode(errors="replace").splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {}


def require_finite(value):
    if isinstance(value, dict):
        for item in value.values():
            require_finite(item)
    elif isinstance(value, list):
        for item in value:
            require_finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError("Non-finite training or evaluation metric")


def validate_training(run, algorithm, budget, num_envs):
    metrics = last_metrics(run)
    require_finite(metrics)
    if (run / "failure.json").exists():
        raise RuntimeError(f"Training reported failure: {run}")
    if metrics.get("vector_steps") != budget * 16:
        raise RuntimeError(f"Incomplete vector-step budget: {metrics}")
    if metrics.get("real_steps") != budget * 16 * num_envs:
        raise RuntimeError("Unexpected real-transition budget")
    if algorithm == "supermimic":
        if metrics.get("policy_updates", 0) <= 0:
            raise RuntimeError("BPTT did not update the Actor")
        if metrics.get("parity_max", {}).get("observation") != 0.0:
            raise RuntimeError("Actor input parity was not verified")
        checkpoint = run / "policy.pt"
        if not (run / "checkpoint_final.pt").is_file():
            raise RuntimeError("Missing final world-model/optimizer checkpoint")
    else:
        candidates = list(run.glob("model_[0-9]*.pt"))
        if not candidates:
            raise RuntimeError("Missing PPO final checkpoint")
        checkpoint = max(candidates, key=lambda p: int(p.stem.split("_")[1]))
        if int(checkpoint.stem.split("_")[1]) != budget - 1:
            raise RuntimeError("PPO checkpoint is not the end-of-budget checkpoint")
    if not checkpoint.is_file():
        raise RuntimeError("Missing policy export")
    return checkpoint, metrics


class Runner:
    def __init__(self, root, repo, env, budget_steps=8000):
        self.root, self.repo, self.env = root, repo, env
        self.budget_steps = budget_steps
        self.python = ["bash", str(repo / "scripts/supermimic/python.sh")]

    def snapshot(self, phase, run, process):
        latest = last_metrics(run)
        remaining = latest.get("estimated_remaining_seconds")
        if remaining is None and latest.get("vector_steps", 0):
            budget = 128 if phase.startswith("smoke") else self.budget_steps
            remaining = latest.get("wall_seconds", 0) * max(0, budget / latest["vector_steps"] - 1)
        try:
            gpu = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                    "--format=csv,noheader",
                ],
                text=True,
                timeout=20,
            ).strip()
        except (OSError, subprocess.SubprocessError) as error:
            gpu = repr(error)
        out = dict(
            checked_utc=stamp(),
            phase=phase,
            pid=process.pid,
            alive=process.poll() is None,
            gpu=gpu,
            latest=latest,
            estimated_remaining_seconds=remaining,
            success_rate_kind="training_clip_completion_fraction; final success comes from 4096 episodes",
        )
        log = self.root / f"{phase}.log"
        if log.exists():
            with log.open("rb") as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 16384))
                lines = stream.read().decode(errors="replace").splitlines()
            out["anomalies"] = [
                line
                for line in lines
                if any(
                    word in line.lower()
                    for word in (
                        "traceback",
                        "out of memory",
                        "segmentation fault",
                        "parity failed",
                    )
                )
            ]
        write_json(self.root / "state.json", out)
        with (self.root / "monitor.jsonl").open("a") as stream:
            stream.write(json.dumps(out) + "\n")
        print(json.dumps(out), flush=True)

    def execute(self, phase, arguments, run):
        command = self.python + [str(item) for item in arguments]
        started = time.monotonic()
        with (self.root / f"{phase}.log").open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=self.repo,
                env=self.env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            write_json(
                self.root / "active_process.json",
                dict(phase=phase, pid=process.pid, command=command, started_utc=stamp()),
            )
            stop = threading.Event()

            def monitor():
                while not stop.wait(3600):
                    try:
                        self.snapshot(phase, run, process)
                    except Exception:
                        with (self.root / "monitor_errors.log").open("a") as stream:
                            stream.write(traceback.format_exc())

            thread = threading.Thread(target=monitor, daemon=True)
            self.snapshot(phase, run, process)
            thread.start()
            try:
                code = process.wait()  # OS wait; health checks occur only hourly.
            finally:
                stop.set()
                thread.join(timeout=30)
            self.snapshot(phase, run, process)
        elapsed = time.monotonic() - started
        write_json(
            self.root / f"{phase}_process.json",
            dict(exit_code=code, wall_seconds=elapsed, command=command),
        )
        if code:
            raise RuntimeError(f"{phase} failed with exit status {code}")
        return elapsed


def setup_case(source, repo, root, canonical, commit, plan_path=None):
    subprocess.run(
        ["git", "clone", "--no-hardlinks", "--quiet", str(source), str(repo)], check=True
    )
    if git(repo, "rev-parse", "HEAD") != commit:
        raise RuntimeError("Clone revision changed")
    for name in ("data", "assets/meshes", "assets/urdf"):
        link = repo / name
        if link.exists():
            raise RuntimeError(f"Resource path should be untracked: {link}")
        link.symlink_to(canonical / name, target_is_directory=True)
    generated = root / "runtime/assets"
    generated.mkdir(parents=True)
    # Canonical is the frozen release checkout; its generated-asset links resolve
    # to the already validated robot assets. Copy them so containers never write
    # a common asset cache. Object USD caches start empty for each case.
    for name in ("usd", "meshes_simplify_collision", "urdf_simplify_collision"):
        original = canonical / "assets" / name
        if not original.is_dir():
            raise RuntimeError(f"Missing prepared robot resource: {original}")
        shutil.copytree(original.resolve(), generated / name, symlinks=False)
    for name in ("usd_cache",):
        (generated / name).mkdir()
    cache = root / "runtime/cache"
    local_cache = (
        Path("/tmp/supermimic-multitask")
        / digest(plan_path or source / PLAN_PATH.relative_to(SOURCE))[:12]
        / root.name
    )
    local_cache.mkdir(parents=True, exist_ok=True)
    cache.symlink_to(local_cache, target_is_directory=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--algorithm", choices=("supermimic", "ppo"), required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    check_source(SOURCE, plan, args.expected_commit)
    task = next(item for item in plan["tasks"] if item["task"] == args.task)
    if digest(task["reference"]) != task["reference_sha256"]:
        raise RuntimeError("Reference data changed")
    args.root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        repo = args.root / "repo"
        setup_case(SOURCE, repo, args.root, args.resources, args.expected_commit, args.plan)
        check_source(repo, plan, args.expected_commit)
        env = os.environ.copy()
        env.update(
            CONTRACK_RESULTS_DIR=str(args.root / "runtime"),
            CONTRACK_ISAACLAB_PATH="/mnt/project/public/jhz/IsaacLab-pinned",
            OMP_NUM_THREADS="4",
            MKL_NUM_THREADS="4",
        )
        runner = Runner(args.root, repo, env, plan["budget_updates"] * 16)
        protocol = dict(
            plan=plan,
            source_commit=args.expected_commit,
            task=task,
            algorithm=args.algorithm,
            initialization="from_scratch",
            created_utc=stamp(),
            cache_policy="Fresh per-case node-local compile cache; smoke compilation reused",
            resource="obstacle_avoidance_4090",
            case_root=str(args.root),
        )
        write_json(args.root / "protocol.json", protocol)
        run = args.root / "run"
        runner.execute(
            "dependencies",
            [
                "-m",
                "pip",
                "install",
                "--no-deps",
                "rsl-rl-lib==3.1.2",
                "tensordict==0.11.0",
                "h5py==3.15.1",
                "trimesh==4.5.1",
                "gymnasium==1.2.1",
                "tensorboard==2.20.0",
            ],
            run,
        )
        runner.execute(
            "runtime_probe",
            [
                "-c",
                "import torch,json; assert torch.cuda.is_available(); "
                'assert torch.__version__.startswith("2.7.0"); '
                "print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,"
                "gpu=torch.cuda.get_device_name(),vram=torch.cuda.get_device_properties(0).total_memory)))",
            ],
            run,
        )
        cfg = json.loads((repo / plan["bptt_config"]).read_text())
        smoke = args.root / "smoke"
        smoke_cfg = cfg | dict(
            budget_updates=8,
            warmup_steps=16,
            wm_only_updates=2,
            warmup_wm_batches=1,
            wm_batches=1,
            log_interval=1,
        )
        write_json(args.root / "smoke_config.json", smoke_cfg)
        base = [
            repo / "scripts/supermimic/train.py",
            "--algorithm",
            args.algorithm,
            "--preset",
            plan["bptt_preset"],
            "--data",
            task["reference"],
            "--embodiment",
            plan.get("embodiment", "xhand"),
            "--seed",
            str(plan["seed"]),
            "--device",
            "cuda:0",
        ]
        smoke_args = base + ["--smoke", "--num-envs", "64", "--run-dir", smoke]
        if args.algorithm == "supermimic":
            smoke_args += ["--config", args.root / "smoke_config.json"]
        runner.execute("smoke", smoke_args, smoke)
        smoke_checkpoint, smoke_metrics = validate_training(smoke, args.algorithm, 8, 64)
        write_json(args.root / "smoke_validated.json", smoke_metrics)
        eval_base = [
            repo / "scripts/supermimic/evaluate.py",
            "--data",
            task["reference"],
            "--embodiment",
            plan.get("embodiment", "xhand"),
            "--seed",
            str(plan["evaluation_seed"]),
            "--device",
            "cuda:0",
        ]
        runner.execute(
            "smoke_export",
            eval_base
            + [
                "--checkpoint",
                smoke_checkpoint,
                "--num-envs",
                "16",
                "--episodes",
                "32",
                "--output",
                args.root / "smoke_evaluation.json",
            ],
            smoke,
        )
        smoke_evaluation = json.loads((args.root / "smoke_evaluation.json").read_text())
        require_finite(smoke_evaluation)
        if (
            smoke_evaluation["episodes"] != 32
            or smoke_evaluation["checkpoint_sha256"] != digest(smoke_checkpoint)
            or smoke_evaluation["reference_sha256"] != task["reference_sha256"]
            or smoke_evaluation["evaluator_git_sha"] != args.expected_commit
        ):
            raise RuntimeError("Smoke export evaluation provenance failed")
        check_source(repo, plan, args.expected_commit)
        train_args = base + [
            "--budget-updates",
            str(plan["budget_updates"]),
            "--num-envs",
            str(plan["num_envs"]),
            "--run-dir",
            run,
        ]
        if args.algorithm == "supermimic":
            train_args += ["--config", repo / plan["bptt_config"]]
        training_wall = runner.execute("training", train_args, run)
        checkpoint, metrics = validate_training(
            run, args.algorithm, plan["budget_updates"], plan["num_envs"]
        )
        manifest = json.loads((run / "manifest.json").read_text())
        if manifest["resume_from"] is not None or manifest["smoke"] or manifest["git_status"]:
            raise RuntimeError("Formal training provenance failed")
        write_json(
            args.root / "training_complete.json",
            dict(
                metrics=metrics,
                training_process_wall_seconds=training_wall,
                checkpoint=str(checkpoint),
                checkpoint_sha256=digest(checkpoint),
            ),
        )
        runner.execute(
            "evaluation",
            eval_base
            + [
                "--checkpoint",
                checkpoint,
                "--num-envs",
                "256",
                "--episodes",
                str(plan["evaluation_episodes"]),
                "--output",
                run / "evaluation.json",
            ],
            run,
        )
        evaluation = json.loads((run / "evaluation.json").read_text())
        require_finite(evaluation)
        if (
            evaluation["episodes"] != plan["evaluation_episodes"]
            or evaluation["checkpoint_sha256"] != digest(checkpoint)
            or evaluation["reference_sha256"] != task["reference_sha256"]
            or evaluation["evaluator_git_sha"] != args.expected_commit
        ):
            raise RuntimeError("Final evaluation provenance failed")
        if args.algorithm == "supermimic":
            runner.execute(
                "wm_evaluation",
                [
                    repo / "scripts/supermimic/evaluate_wm.py",
                    run / "checkpoint_final.pt",
                    "--device",
                    "cuda:0",
                    "--output",
                    run / "wm_evaluation.json",
                ],
                run,
            )
            require_finite(json.loads((run / "wm_evaluation.json").read_text()))
        check_source(repo, plan, args.expected_commit)
        write_json(
            args.root / "completion.json",
            dict(
                state="evaluated",
                completed_utc=stamp(),
                task=args.task,
                algorithm=args.algorithm,
                initialization="from_scratch",
                source_commit=args.expected_commit,
                evaluation=evaluation,
                training_process_wall_seconds=training_wall,
                end_to_end_wall_seconds=time.monotonic() - started,
            ),
        )
    except BaseException:
        write_json(
            args.root / "failure.json",
            dict(
                failed_utc=stamp(),
                traceback=traceback.format_exc(),
                task=args.task,
                algorithm=args.algorithm,
            ),
        )
        write_json(
            args.root / "completion.json",
            dict(
                state="failed",
                completed_utc=stamp(),
                task=args.task,
                algorithm=args.algorithm,
                end_to_end_wall_seconds=time.monotonic() - started,
            ),
        )
        raise


if __name__ == "__main__":
    main()
