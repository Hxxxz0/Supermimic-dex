from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from . import resolve_asset_path

__all__ = [
    "XARM_XHAND_RIGHT_CFG",
    "XARM_JOINT_NAMES",
    "HAND_JOINT_NAMES",
    "DEFAULT_XARM_QPOS",
]

_USD_PATH = resolve_asset_path("usd", "xarm_xhand_right", "xarm_xhand_right.usd")

DEFAULT_XARM_QPOS = [0.0, 0.0, 0.0, 0.22, 3.14, 1.33, 0.0]
XARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
HAND_JOINT_NAMES = [
    "right_hand_thumb_bend_joint",
    "right_hand_thumb_rota_joint1",
    "right_hand_thumb_rota_joint2",
    "right_hand_index_bend_joint",
    "right_hand_index_joint1",
    "right_hand_index_joint2",
    "right_hand_mid_joint1",
    "right_hand_mid_joint2",
    "right_hand_ring_joint1",
    "right_hand_ring_joint2",
    "right_hand_pinky_joint1",
    "right_hand_pinky_joint2",
]

_DEFAULT_JOINT_STATE: dict[str, float] = {
    name: value for name, value in zip(XARM_JOINT_NAMES, DEFAULT_XARM_QPOS, strict=True)
}
_DEFAULT_JOINT_STATE.update({name: 0.0 for name in HAND_JOINT_NAMES})

XARM_XHAND_RIGHT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(_USD_PATH),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos=_DEFAULT_JOINT_STATE,
        pos=(0.0, 0.0, 0.0),
        # Compensate the 180° yaw introduced by USD conversion (fix_base) so the hand
        # stays on the +X side of the arm in the stage/world frame.
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "xarm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-7]"],
            effort_limit_sim=50.0,
            stiffness=1000.0,
            damping=50.0,
        ),
        "xhand": ImplicitActuatorCfg(
            joint_names_expr=["right_hand_.*joint.*"],
            effort_limit_sim=1.0,
            stiffness=100.0,
            damping=10.0,
        ),
    },
)
