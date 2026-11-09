from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

__all__ = ["get_assets_root", "resolve_asset_path"]

_ASSETS_ENV_VAR = "CONTRACK_ASSETS_ROOT"


def _detect_repo_root() -> Path:
    """Walk upwards from this file until an assets directory is found."""

    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "assets").exists():
            return parent
    raise FileNotFoundError(
        "Unable to locate the project root that contains an assets directory. "
        "Set the CONTRACK_ASSETS_ROOT environment variable to override the detection."
    )


def get_assets_root() -> Path:
    """Return the absolute path to the repository's assets directory."""

    env_path = os.environ.get(_ASSETS_ENV_VAR)
    if env_path:
        resolved = Path(env_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(
                f"CONTRACK_ASSETS_ROOT points to '{resolved}', but the path does not exist."
            )
        return resolved

    repo_root = _detect_repo_root()
    assets_dir = repo_root / "assets"
    if not assets_dir.exists():
        raise FileNotFoundError(
            f"Detected project root '{repo_root}', but the assets directory is missing. "
            "Set CONTRACK_ASSETS_ROOT to the correct location."
        )
    return assets_dir


def resolve_asset_path(*relative_parts: Iterable[str] | str) -> Path:
    """Shortcut to join entries inside the assets directory."""

    path = get_assets_root()
    for part in relative_parts:
        if isinstance(part, str):
            path = path / part
        else:
            path = path.joinpath(*part)
    return path
