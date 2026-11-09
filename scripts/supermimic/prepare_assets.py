#!/usr/bin/env python3
"""Prepare the three official robot assets in one headless Isaac session."""

import argparse
import os
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main():
    from isaaclab.app import AppLauncher
    from contrack_supermimic.experiment import RESULT_ROOT, prepare_output_links

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    prepare_output_links(REPO)
    subprocess.run([sys.executable, str(REPO / "assets/simplify_mesh.py")], cwd=REPO, check=True)
    args.headless, args.enable_cameras = True, False
    app = AppLauncher(args).app
    try:
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
        from pxr import Usd, UsdGeom

        for name in ("xarm_xhand_left", "xarm_xhand_right", "rby1a_sharpa"):
            source = (
                REPO
                / "assets"
                / ("urdf" if name == "rby1a_sharpa" else "urdf_simplify_collision")
                / (name + ("" if name == "rby1a_sharpa" else "-simplified") + ".urdf")
            )
            target = RESULT_ROOT / "assets/usd" / name / f"{name}.usd"
            if target.exists() and not args.force:
                print(f"Already prepared {target}", flush=True)
                continue
            UrdfConverter(
                UrdfConverterCfg(
                    asset_path=str(source.resolve()),
                    usd_dir=str(target.parent),
                    usd_file_name=target.name,
                    force_usd_conversion=True,
                    make_instanceable=True,
                    fix_base=True,
                    merge_fixed_joints=True,
                    joint_drive=UrdfConverterCfg.JointDriveCfg(
                        target_type="none",
                        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
                    ),
                )
            )
            stage = Usd.Stage.Open(str(target.parent / "configuration" / f"{name}_physics.usd"))
            UsdGeom.Xform.Define(stage, "/visuals")
            for link in ET.parse(source).getroot().iter("link"):
                UsdGeom.Xform.Define(stage, f"/visuals/{link.attrib['name']}")
            stage.GetRootLayer().Save()
            print(f"Prepared {target}", flush=True)
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(1)
    finally:
        app.close()


if __name__ == "__main__":
    main()
