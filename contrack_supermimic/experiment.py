import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


RESULT_ROOT = (
    Path(
        os.environ.get("CONTRACK_RESULTS_DIR", str(Path(__file__).resolve().parents[1] / "outputs"))
    )
    .expanduser()
    .resolve()
)
UPSTREAM = "f5efe18e4b1f986566b252265c4bf63f3b67c6c6"
DONOR = "2becb7380fed86171056f9bf4338b2a6a0c14b01"
ISAACLAB = "42e61645c96bac08135566634785cdc87728d5ab"


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def runtime_fingerprint():
    packages = {}
    for name in ("torch", "isaaclab", "isaaclab_rl", "rsl-rl-lib", "tensordict", "h5py"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    versions = {
        str(p): p.read_text().strip()
        for p in (Path("/isaac-sim/VERSION"), Path("/workspace/isaaclab/VERSION"))
        if p.exists()
    }
    try:
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        gpu = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "versions": versions,
        "gpu": gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "image_digest": os.environ.get("CONTRACK_IMAGE_DIGEST"),
    }


def manifest(
    repo,
    data,
    algorithm,
    config,
    num_envs,
    seed,
    run_dir,
    resume=None,
    smoke=False,
    source_repo=None,
):
    dirty = git(repo, "status", "--porcelain")
    if dirty and not smoke:
        raise RuntimeError(
            "Formal experiments require committed source; use --smoke for development"
        )
    assets = {}
    for p in (Path(repo) / "assets" / "usd").rglob("*.usd"):
        assets[str(p.relative_to(repo))] = sha256(p)
    for p in (Path(repo) / "assets" / "usd_cache" / Path(data).stem).glob("*.usd"):
        assets[str(p.relative_to(repo))] = sha256(p)
    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": algorithm,
        "git_sha": git(repo, "rev-parse", "HEAD"),
        "git_status": dirty,
        "upstream_sha": UPSTREAM,
        "donor_sha": DONOR,
        "isaaclab_sha": ISAACLAB,
        "runtime": runtime_fingerprint(),
        "reference_path": str(Path(data).resolve()),
        "reference_sha256": sha256(data),
        "asset_sha256": assets,
        "num_envs": num_envs,
        "seed": seed,
        "config": config,
        "resume_from": str(Path(resume).resolve()) if resume else None,
        "smoke": smoke,
    }
    source_repo = source_repo or repo
    out.update(source_repo=str(source_repo), source_git_sha=git(source_repo, "rev-parse", "HEAD"))
    (Path(run_dir) / "manifest.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def prepare_output_links(repo):
    """Redirect upstream-generated files through ignored symlinks, without source edits."""
    repo = Path(repo)
    for name in ("usd", "usd_cache", "meshes_simplify_collision", "urdf_simplify_collision"):
        target = RESULT_ROOT / "assets" / name
        target.mkdir(parents=True, exist_ok=True)
        link = repo / "assets" / name
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise RuntimeError(f"Unexpected output link: {link}")
        elif link.exists():
            raise RuntimeError(f"Refusing to replace existing output directory {link}")
        else:
            link.symlink_to(target, target_is_directory=True)
    # Converted URDF mesh paths are relative to the shared asset directory.
    meshes = RESULT_ROOT / "assets" / "meshes"
    if not meshes.exists():
        meshes.symlink_to(repo / "assets" / "meshes", target_is_directory=True)
    log_target = RESULT_ROOT / "runs" / "upstream_logs"
    log_target.mkdir(parents=True, exist_ok=True)
    log_link = repo / "logs"
    if not log_link.exists():
        log_link.symlink_to(log_target, target_is_directory=True)


class PPOAudit:
    """Common terminal-state timing/metrics for the unchanged upstream PPO algorithm."""

    def __init__(self, env, adapter, run_dir, interval=256):
        import time

        self.env, self.adapter, self.run_dir = env, adapter, Path(run_dir)
        self.interval, self.steps, self.started = interval, 0, time.perf_counter()
        self.adapter.attach()
        self.rewards, self.completions, self.episodes = [], 0, 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        self.adapter.begin()
        out = self.env.step(action)
        record = self.adapter.finish()
        self.rewards.append(record["channels"].mean(0))
        self.completions += (record["timeout"] & ~record["terminated"]).sum()
        self.episodes += record["done"].sum()
        self.steps += 1
        if self.steps % self.interval == 0:
            self.log()
        return out

    def log(self):
        import torch
        import time

        if not self.rewards:
            return
        reward = torch.stack(self.rewards).mean(0).tolist()
        row = {
            "equivalent_ppo_iteration": self.steps / 16,
            "vector_steps": self.steps,
            "real_steps": self.steps * self.env.num_envs,
            "wall_seconds": time.perf_counter() - self.started,
            "rg": reward[0],
            "rs": reward[1],
            "rp": reward[2],
            "training_clip_completion_fraction": float(
                self.completions / max(1, int(self.episodes))
            ),
        }
        with (self.run_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        self.rewards, self.completions, self.episodes = [], 0, 0
