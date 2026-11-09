"""MDP terms for RB-Y1A+Sharpa HDF5 physics playback."""

from ConTrack.tasks.manager_based.mimic_rby1a_sharpa.mdp.actions import (
    H5Rby1aSharpaPlaybackAction,
    H5Rby1aSharpaPlaybackActionCfg,
    Rby1aSharpaResidualAction,
    Rby1aSharpaResidualActionCfg,
)
from ConTrack.tasks.manager_based.mimic_rby1a_sharpa.mdp.observations import (
    all_qpos,
    all_qvel,
    delayed_all_qpos,
    delayed_all_qvel,
    phase,
    ref_all_qpos,
    ref_hand_contact,
    hand_contact,
)
from ConTrack.tasks.manager_based.mimic_rby1a_sharpa.mdp.rewards import (
    contact_distance_reward,
    contact_reward,
    qacc_penalty_arm,
    qacc_penalty_finger,
    qacc_penalty_obj_pos,
    qacc_penalty_obj_rot,
    qvel_penalty_arm,
    qvel_penalty_finger,
    qvel_penalty_obj_pos,
    qvel_penalty_obj_rot,
    tracking_arm,
    tracking_finger,
    tracking_obj_pos,
    tracking_obj_rot,
)
from ConTrack.tasks.manager_based.mimic_rby1a_sharpa.mdp.domain_randomization import (
    perturb_objects_xy,
    randomize_joint_pd_gains,
)
from ConTrack.tasks.manager_based.mimic_rby1a_sharpa.mdp.state import (
    advance_reference_frame,
    filter_sharpa_self_collisions,
    init_reference_buffers,
    reset_reference_episode,
    reset_reference_episode_first_frame,
)
from ConTrack.tasks.manager_based.mimic_rby1a_sharpa.mdp.terminations import (
    obj_pose_exceeded,
    time_out,
)

__all__ = [
    "H5Rby1aSharpaPlaybackAction",
    "H5Rby1aSharpaPlaybackActionCfg",
    "Rby1aSharpaResidualAction",
    "Rby1aSharpaResidualActionCfg",
    "advance_reference_frame",
    "all_qpos",
    "all_qvel",
    "delayed_all_qpos",
    "delayed_all_qvel",
    "contact_distance_reward",
    "contact_reward",
    "init_reference_buffers",
    "filter_sharpa_self_collisions",
    "obj_pose_exceeded",
    "phase",
    "perturb_objects_xy",
    "randomize_joint_pd_gains",
    "qacc_penalty_arm",
    "qacc_penalty_finger",
    "qacc_penalty_obj_pos",
    "qacc_penalty_obj_rot",
    "qvel_penalty_arm",
    "qvel_penalty_finger",
    "qvel_penalty_obj_pos",
    "qvel_penalty_obj_rot",
    "ref_all_qpos",
    "ref_hand_contact",
    "reset_reference_episode",
    "reset_reference_episode_first_frame",
    "time_out",
    "tracking_arm",
    "tracking_finger",
    "tracking_obj_pos",
    "tracking_obj_rot",
    "hand_contact",
]

