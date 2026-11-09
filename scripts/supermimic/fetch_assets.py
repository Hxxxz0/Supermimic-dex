#!/usr/bin/env python3
"""Fetch hash-pinned ConTrack robot resources and the Hammer reference clip."""

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(repo=REPO, archive_path=None, verify_only=False):
    repo = Path(repo)
    manifest = json.loads((repo / "configs/upstream_resources.json").read_text())
    files = manifest["files"]
    missing = [
        name
        for name, item in files.items()
        if not (repo / name).is_file() or sha256(repo / name) != item["sha256"]
    ]
    if not missing:
        print(f"Verified {len(files)} upstream resource files.")
        return
    if verify_only:
        raise ValueError(
            f"{len(missing)} missing or mismatched resources; run fetch_assets.py first"
        )
    results = Path(os.environ.get("CONTRACK_RESULTS_DIR", str(repo / "outputs"))).expanduser()
    cache = results / "cache/downloads"
    cache.mkdir(parents=True, exist_ok=True)
    if archive_path is None:
        archive_path = cache / f"ConTrack-{manifest['upstream_sha']}.tar.gz"
        if not archive_path.is_file():
            url = f"https://codeload.github.com/EmptyBlueBox/ConTrack/tar.gz/{manifest['upstream_sha']}"
            print(f"Downloading pinned ConTrack resources: {manifest['upstream_sha']}", flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "Supermimic-dex/0.1"})
            with tempfile.NamedTemporaryFile(dir=cache, suffix=".tmp", delete=False) as target:
                temporary = Path(target.name)
                try:
                    with urllib.request.urlopen(request, timeout=120) as response:
                        shutil.copyfileobj(response, target)
                    target.close()
                    temporary.replace(archive_path)
                finally:
                    temporary.unlink(missing_ok=True)
    prefix = f"ConTrack-{manifest['upstream_sha']}/"
    with tarfile.open(archive_path, "r:*") as archive:
        for name in missing:
            # Only manifest-listed regular files are copied; archive paths are never extracted.
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe resource path: {name}")
            destination = repo / relative
            if not destination.resolve().is_relative_to(repo.resolve()):
                raise ValueError(f"Resource destination leaves repository: {name}")
            member = archive.getmember(prefix + name)
            if not member.isfile() or member.size != files[name]["bytes"]:
                raise ValueError(f"Unexpected resource type or size: {name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Unreadable archive member: {name}")
            with (
                stream,
                tempfile.NamedTemporaryFile(
                    dir=destination.parent, suffix=".tmp", delete=False
                ) as target,
            ):
                temporary = Path(target.name)
                try:
                    shutil.copyfileobj(stream, target)
                    target.close()
                    if sha256(temporary) != files[name]["sha256"]:
                        raise ValueError(f"Resource hash mismatch: {name}")
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
    print(
        f"Fetched and verified {len(missing)} files; all {len(files)} resources match the pinned source."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use an already downloaded upstream archive")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    fetch(archive_path=args.archive, verify_only=args.verify_only)


if __name__ == "__main__":
    main()
