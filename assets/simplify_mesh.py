#!/usr/bin/env python3
"""
Replace XHand collision meshes with convex hulls and drop the rest.

This script writes and updates URDFs in ``assets/urdf_simplify_collision``:
- Copies base URDFs from ``assets/urdf`` into ``assets/urdf_simplify_collision`` as
  ``*-simplified.urdf`` (overwrites on every run).
- Keeps collision meshes only for 12 XHand links per hand:
  - ``{left,right}_hand_link`` (palm)
  - ``{left,right}_hand_index_bend_link``
  - ``{left,right}_hand_*_link1``
  - ``{left,right}_hand_*_link2``
- Drops collisions for links such as ``*rotaback*``, ``*back*``, and ``*tip*``,
  and drops all non-hand collisions (e.g. arm collision meshes in combined URDFs).
- Replaces the kept collisions with convex-hull meshes exported to
  ``assets/meshes_simplify_collision``.

Usage (from the repo root):
    python assets/simplify_mesh.py
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import trimesh


def keep_collision_link(link_name: str) -> bool:
    """Return whether a link should keep collision geometry after simplification.

    Parameters
    ----------
    link_name : str
        URDF link name.

    Returns
    -------
    keep : bool
        True if ``link_name`` is the palm (``left_hand_link``/``right_hand_link``)
        or matches ``{left,right}_hand_*_link{1,2}`` excluding any name containing
        ``"back"`` or ``"rotaback"``.
    """

    if link_name in ("left_hand_link", "right_hand_link"):
        return True
    if link_name in ("left_hand_index_bend_link", "right_hand_index_bend_link"):
        return True
    if not (link_name.startswith("left_hand_") or link_name.startswith("right_hand_")):
        return False
    if "rotaback" in link_name or "back" in link_name:
        return False
    return link_name.endswith("_link1") or link_name.endswith("_link2")


def simplify_urdf(
    urdf_path: Path, out_mesh_dir: Path
) -> None:
    """Simplify a URDF by keeping only XHand collisions and converting them to hull meshes.

    The convex hull is computed from each kept link's ``visual`` mesh.

    Parameters
    ----------
    urdf_path : Path
        Input/output URDF path. The file is rewritten in-place.
    out_mesh_dir : Path
        Directory where convex hull meshes are written. Output paths preserve the
        subpath under ``assets/meshes`` (e.g. ``xhand/right_hand_link.STL``).

    Returns
    -------
    None
    """

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    names = [(link.get("name") or "") for link in root.findall("link")]
    if not any(
        token in name
        for name in names
        for token in ("_hand_thumb", "_hand_index", "_hand_mid", "_hand_ring", "_hand_pinky")
    ):
        return

    mesh_root = Path(__file__).resolve().parent / "meshes"
    hull_written = 0
    kept_collisions = 0

    for link in root.findall("link"):
        link_name = link.get("name", "")
        for collision in list(link.findall("collision")):
            link.remove(collision)
        if not keep_collision_link(link_name):
            continue

        visual_mesh_node = link.find("visual/geometry/mesh")
        mesh_path = (urdf_path.parent / Path(visual_mesh_node.get("filename"))).resolve()
        hull_path = out_mesh_dir / mesh_path.relative_to(mesh_root)
        if not hull_path.exists():
            mesh = trimesh.load_mesh(mesh_path, force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(tuple(mesh.dump().values()))
            hull_path.parent.mkdir(parents=True, exist_ok=True)
            mesh.convex_hull.export(hull_path)
            hull_written += 1

        visual_origin = link.find("visual/origin")
        collision = ET.SubElement(link, "collision")
        origin = ET.SubElement(collision, "origin")
        origin.set("xyz", (visual_origin.get("xyz") if visual_origin is not None else "0 0 0"))
        origin.set("rpy", (visual_origin.get("rpy") if visual_origin is not None else "0 0 0"))
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(
            geometry,
            "mesh",
            filename=Path(os.path.relpath(hull_path, urdf_path.parent)).as_posix(),
        )
        kept_collisions += 1

    ET.indent(tree, space="  ")
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    print(f"[ok] {urdf_path.name} (kept={kept_collisions}, hulls={hull_written})")


def main() -> None:
    """Run the simplification pipeline over ``assets/urdf_simplify_collision``.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    root = Path(__file__).resolve().parent
    src_urdf_dir = root / "urdf"
    urdf_dir = root / "urdf_simplify_collision"
    out_mesh_dir = root / "meshes_simplify_collision"
    urdf_dir.mkdir(parents=True, exist_ok=True)
    for stem in ("xarm_xhand_left", "xarm_xhand_right", "xhand_left", "xhand_right"):
        shutil.copyfile(src_urdf_dir / f"{stem}.urdf", urdf_dir / f"{stem}-simplified.urdf")
    for urdf_path in sorted(urdf_dir.glob("*.urdf")):
        simplify_urdf(urdf_path, out_mesh_dir)


if __name__ == "__main__":
    main()
