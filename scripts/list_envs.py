# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to print all the available environments in Isaac Lab.

The script iterates over all registered environments and stores the details in a table.
It prints the name of the environment, the entry point and the config file.

All the environments are registered in the `ConTrack` extension. They start
with `Isaac` in their name.
"""

"""Launch Isaac Sim Simulator first."""

import sys
from pathlib import Path

from isaaclab.app import AppLauncher

repo_root = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if p not in ("", str(repo_root))]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "source" / "ConTrack"))

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import ConTrack.tasks  # noqa: F401
from prettytable import PrettyTable


def main():
    """Print all environments registered in `ConTrack` extension."""
    # print all the available environments
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Environments in Isaac Lab"
    # set alignment of table columns
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    # count of environments
    index = 0
    # acquire all Isaac environments names
    for task_spec in gym.registry.values():
        if task_spec.id.startswith("Isaac-") and (
            "Xarm-Xhand-Mimic" in task_spec.id
            or "Rby1a-Sharpa-Mimic" in task_spec.id
        ):
            # add details to table
            table.add_row(
                [
                    index + 1,
                    task_spec.id,
                    task_spec.entry_point,
                    task_spec.kwargs["env_cfg_entry_point"],
                ]
            )
            # increment count
            index += 1

    print(table)


if __name__ == "__main__":
    try:
        # run the main function
        main()
    except Exception as e:
        raise e
    finally:
        # close the app
        simulation_app.close()
