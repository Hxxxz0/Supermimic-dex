from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from ConTrack.robots import resolve_asset_path

__all__ = [
    "RBY1A_SHARPA_CFG",
    "RBY1A_LEFT_JOINT_NAMES",
    "RBY1A_RIGHT_JOINT_NAMES",
    "SHARPA_LEFT_JOINT_NAMES",
    "SHARPA_RIGHT_JOINT_NAMES",
    "DEFAULT_RBY1A_QPOS",
]

_USD_PATH = resolve_asset_path("usd", "rby1a_sharpa", "rby1a_sharpa.usd")

DEFAULT_RBY1A_QPOS = [0.0, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0]
RBY1A_RIGHT_JOINT_NAMES = [f"right_arm_{i}" for i in range(7)]
RBY1A_LEFT_JOINT_NAMES = [f"left_arm_{i}" for i in range(7)]

SHARPA_RIGHT_JOINT_NAMES = [
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
]
SHARPA_LEFT_JOINT_NAMES = [
    s.replace("right_", "left_") for s in SHARPA_RIGHT_JOINT_NAMES
]

_SHARPA_RIGHT_CALIB = {
    "right_thumb_CMC_FE": (0.12138, 0.004206, 0.0032, 0.132, 678.426),
    "right_thumb_CMC_AA": (0.2304, 0.0078826, 0.0032, 0.132, 678.426),
    "right_thumb_MCP_FE": (0.083078, 0.003194, 0.00265, 0.104, 921.14),
    "right_thumb_MCP_AA": (0.11557, 0.00363, 0.00265, 0.104, 921.14),
    "right_thumb_IP": (0.01584, 0.000698, 0.00061, 0.02476, 665.6805),
    "right_index_MCP_FE": (0.083078, 0.003194, 0.00265, 0.104, 921.14),
    "right_index_MCP_AA": (0.11557, 0.00363, 0.00265, 0.104, 921.14),
    "right_index_PIP": (0.01584, 0.000698, 0.00061, 0.02476, 665.6805),
    "right_index_DIP": (0.01578, 0.00055, 0.00042, 0.000418, 840.2969),
    "right_middle_MCP_FE": (0.083078, 0.003194, 0.00265, 0.104, 921.14),
    "right_middle_MCP_AA": (0.11557, 0.00363, 0.00265, 0.104, 921.14),
    "right_middle_PIP": (0.01584, 0.000698, 0.00061, 0.02476, 665.6805),
    "right_middle_DIP": (0.01578, 0.00055, 0.00042, 0.000418, 840.2969),
    "right_ring_MCP_FE": (0.083078, 0.003194, 0.00265, 0.104, 921.14),
    "right_ring_MCP_AA": (0.11557, 0.00363, 0.00265, 0.104, 921.14),
    "right_ring_PIP": (0.01584, 0.000698, 0.00061, 0.02476, 665.6805),
    "right_ring_DIP": (0.01578, 0.00055, 0.00042, 0.000418, 840.2969),
    "right_pinky_CMC": (0.02409, 0.000685, 0.00012, 0.013, 2009.6014),
    "right_pinky_MCP_FE": (0.083078, 0.003194, 0.00265, 0.104, 921.14),
    "right_pinky_MCP_AA": (0.11557, 0.00363, 0.00265, 0.104, 921.14),
    "right_pinky_PIP": (0.01584, 0.000698, 0.00061, 0.02476, 665.6805),
    "right_pinky_DIP": (0.01578, 0.00055, 0.00042, 0.000418, 840.2969),
}
_SHARPA_RIGHT_EFFORT_LIMIT_SIM = {
    "right_thumb_CMC_FE": 3.3,
    "right_thumb_CMC_AA": 3.3,
    "right_thumb_MCP_FE": 1.864,
    "right_thumb_MCP_AA": 1.864,
    "right_thumb_IP": 0.638,
    "right_index_MCP_FE": 1.864,
    "right_index_MCP_AA": 1.864,
    "right_index_PIP": 0.638,
    "right_index_DIP": 0.189369,
    "right_middle_MCP_FE": 1.864,
    "right_middle_MCP_AA": 1.864,
    "right_middle_PIP": 0.638,
    "right_middle_DIP": 0.189369,
    "right_ring_MCP_FE": 1.864,
    "right_ring_MCP_AA": 1.864,
    "right_ring_PIP": 0.638,
    "right_ring_DIP": 0.189369,
    "right_pinky_CMC": 0.5285,
    "right_pinky_MCP_FE": 1.864,
    "right_pinky_MCP_AA": 1.864,
    "right_pinky_PIP": 0.638,
    "right_pinky_DIP": 0.189369,
}
_SHARPA_CALIB = _SHARPA_RIGHT_CALIB | {
    k.replace("right_", "left_"): v for k, v in _SHARPA_RIGHT_CALIB.items()
}
_SHARPA_EFFORT_LIMIT_SIM = _SHARPA_RIGHT_EFFORT_LIMIT_SIM | {
    k.replace("right_", "left_"): v for k, v in _SHARPA_RIGHT_EFFORT_LIMIT_SIM.items()
}
_RAD_TO_DEG = 180.0 / math.pi
_DEG_TO_RAD = math.pi / 180.0
_SHARPA_STIFFNESS = {k: v[0] * _RAD_TO_DEG for k, v in _SHARPA_CALIB.items()}
_SHARPA_DAMPING = {k: v[1] * _RAD_TO_DEG for k, v in _SHARPA_CALIB.items()}
_SHARPA_ARMATURE = {k: v[2] for k, v in _SHARPA_CALIB.items()}
_SHARPA_FRICTION = {k: v[3] for k, v in _SHARPA_CALIB.items()}
_SHARPA_VELOCITY_LIMIT_SIM = {k: v[4] * _DEG_TO_RAD for k, v in _SHARPA_CALIB.items()}

_DEFAULT_JOINT_STATE = {}
for _names in (RBY1A_LEFT_JOINT_NAMES, RBY1A_RIGHT_JOINT_NAMES):
    _DEFAULT_JOINT_STATE.update(
        {name: value for name, value in zip(_names, DEFAULT_RBY1A_QPOS, strict=True)}
    )
_DEFAULT_JOINT_STATE.update(
    {name: 0.0 for name in SHARPA_LEFT_JOINT_NAMES + SHARPA_RIGHT_JOINT_NAMES}
)

RBY1A_SHARPA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(_USD_PATH),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,
            enabled_self_collisions=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos=_DEFAULT_JOINT_STATE,
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "rby1a": ImplicitActuatorCfg(
            joint_names_expr=["left_arm_[0-6]", "right_arm_[0-6]"],
            effort_limit_sim=50.0,
            stiffness=1000.0,
            damping=50.0,
        ),
        "sharpa": ImplicitActuatorCfg(
            joint_names_expr=SHARPA_LEFT_JOINT_NAMES + SHARPA_RIGHT_JOINT_NAMES,
            effort_limit_sim=_SHARPA_EFFORT_LIMIT_SIM,
            stiffness=_SHARPA_STIFFNESS,
            damping=_SHARPA_DAMPING,
            armature=_SHARPA_ARMATURE,
            friction=_SHARPA_FRICTION,
            velocity_limit_sim=_SHARPA_VELOCITY_LIMIT_SIM,
        ),
    },
)

