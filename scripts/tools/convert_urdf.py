"""Convert a URDF to a USD using Isaac Lab's UrdfConverter.

Parameters
----------
urdf_path : str
    Absolute path to the source URDF file.
usd_path : str
    Absolute path to the output USD file (e.g. ``.../xhand_left.usd``).
--fix-base : bool
    If set, fixes the root link to the world frame.
--merge-joints : bool
    If set, merges fixed joints during import.
--joint-target-type : str
    Joint drive target type in ``{"none", "position", "velocity"}``.
--joint-stiffness : float
    Joint drive stiffness applied during URDF import.
--joint-damping : float
    Joint drive damping applied during URDF import.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert URDF to USD (Isaac Lab).")
parser.add_argument("urdf_path", type=str)
parser.add_argument("usd_path", type=str)
parser.add_argument("--fix-base", action="store_true")
parser.add_argument("--merge-joints", action="store_true")
parser.add_argument(
    "--joint-target-type",
    type=str,
    default="position",
    choices=("none", "position", "velocity"),
)
parser.add_argument("--joint-stiffness", type=float, default=0.0)
parser.add_argument("--joint-damping", type=float, default=0.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from pxr import Usd, UsdGeom

urdf_path = str(Path(args_cli.urdf_path).resolve())
usd_path = Path(args_cli.usd_path).resolve()
UrdfConverter(
    UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=str(usd_path.parent),
        usd_file_name=usd_path.name,
        force_usd_conversion=True,
        make_instanceable=True,
        fix_base=args_cli.fix_base,
        merge_fixed_joints=args_cli.merge_joints,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type=args_cli.joint_target_type,
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=args_cli.joint_stiffness, damping=args_cli.joint_damping
            ),
        ),
    )
)

physics_usd_path = usd_path.parent / "configuration" / f"{usd_path.stem}_physics.usd"
stage = Usd.Stage.Open(str(physics_usd_path))
UsdGeom.Xform.Define(stage, "/visuals")
for link in ET.parse(urdf_path).getroot().iter("link"):
    UsdGeom.Xform.Define(stage, f"/visuals/{link.attrib['name']}")
stage.GetRootLayer().Save()
simulation_app.close()
