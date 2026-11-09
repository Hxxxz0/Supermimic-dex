"""Resource publication must fail safely on corrupt archives and destination escapes."""

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest


def resource_fixture(tmp_path, *, tampered=False, symlink=False):
    path = Path(__file__).resolve().parents[2] / "scripts/supermimic/fetch_assets.py"
    spec = importlib.util.spec_from_file_location("release_assets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    payload = b"pinned upstream bytes"
    manifest = {
        "upstream_sha": "test",
        "files": {
            "data/clip.h5": {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        },
    }
    (repo / "configs/upstream_resources.json").write_text(json.dumps(manifest))
    archive_path = tmp_path / "resources.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo("ConTrack-test/data/clip.h5")
        member.size = len(payload)
        if symlink:
            member.type = tarfile.SYMTYPE
            member.linkname = "../../outside"
            archive.addfile(member)
        else:
            archive.addfile(member, io.BytesIO(b"x" * len(payload) if tampered else payload))
    return module, repo, archive_path, payload


def test_pinned_resource_download_and_repeat_verification(tmp_path):
    module, repo, archive, payload = resource_fixture(tmp_path)
    module.fetch(repo, archive)
    assert (repo / "data/clip.h5").read_bytes() == payload
    module.fetch(repo, verify_only=True)


def test_corrupt_resource_never_replaces_destination(tmp_path):
    module, repo, archive, _ = resource_fixture(tmp_path, tampered=True)
    with pytest.raises(ValueError, match="hash mismatch"):
        module.fetch(repo, archive)
    assert not (repo / "data/clip.h5").exists()
    assert not list((repo / "data").glob("*.tmp"))


def test_archive_symlink_cannot_escape_extraction(tmp_path):
    module, repo, archive, _ = resource_fixture(tmp_path, symlink=True)
    with pytest.raises(ValueError, match="type or size"):
        module.fetch(repo, archive)
    assert not (repo / "data/clip.h5").exists()


def test_existing_destination_symlink_cannot_write_outside_repo(tmp_path):
    module, repo, archive, _ = resource_fixture(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (repo / "data").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="leaves repository"):
        module.fetch(repo, archive)
    assert not (outside / "clip.h5").exists()
