# Third Party Notices

This repository contains code, assets, and data with separate upstream sources.

## Isaac Lab

Source

https://github.com/isaac-sim/IsaacLab

License

BSD 3 Clause License

The ConTrack Isaac Lab extension structure and selected script patterns are adapted from Isaac Lab.

## RSL RL

Source

https://github.com/leggedrobotics/rsl_rl

License

BSD 3 Clause License

The `rsl_rl_contrack` package is based on RSL RL.

## xArm

Sources

https://www.ufactory.cc

https://github.com/xArm-Developer/xarm_ros

License

The xArm ROS repository is distributed under the BSD 3 Clause License. xArm robot assets are attributed to UFACTORY official resources.

## XHAND1

Source

https://www.robotera.com/

https://www.robotera.com/download.html

License

XHAND1 assets are attributed to RobotEra official resources. Separate RobotEra asset terms may apply.

## RB-Y1A Assets

Sources

https://github.com/RainbowRobotics/rby1-sdk

https://rainbowrobotics.github.io/rby1-dev/

License

Apache License 2.0

Copyright 2024-2025 Rainbow Robotics.

The RB-Y1A portions of `assets/urdf/rby1a_sharpa.urdf`, the meshes under `assets/meshes/rby1a`, and the corresponding generated USD assets are derived from the official Rainbow Robotics RB-Y1 model. ConTrack integrates the RB-Y1A model with dual Sharpa hands and adds simulator-specific articulation, actuator, collision, and physics configuration. The resulting files are modified derivatives and are not official Rainbow Robotics releases. RB-Y1 and Rainbow Robotics names and marks remain the property of Rainbow Robotics, and this attribution does not imply endorsement.

## Sharpa Assets

Sources

https://www.sharpa.com/pages/downloads

https://github.com/sharpa-robotics/sharpa-urdf-usd-xml

License

Apache License 2.0

Copyright 2025 Sharpa Group.

The hand descriptions under `assets/urdf/sharpa_left.urdf` and `assets/urdf/sharpa_right.urdf`, the meshes under `assets/meshes/sharpa`, the Sharpa portions of `assets/urdf/rby1a_sharpa.urdf`, and the corresponding generated USD assets are derived from official Sharpa hardware assets. ConTrack integrates the left and right hand models with RB-Y1A and adds simulator-specific physical parameters, actuator configuration, contact sensing, and collision filtering. The resulting files are modified derivatives and are not official Sharpa releases. Sharpa names and marks remain the property of Sharpa Group, and this attribution does not imply endorsement.

## DexterHand

Source

https://huggingface.co/datasets/pku-mocca/DexterHand

License

Apache 2.0

The included HDF5 examples under `data/xhand` are derived from DexterHand demonstrations.

## ARCTIC

Sources

https://arctic.is.tue.mpg.de/

https://github.com/zc-alexfan/arctic

License

Data & Software Copyright License for non-commercial scientific research purposes

The files matching `data/xhand/arctic-*.h5` and `data/sharpa/arctic-*.h5` are unofficial derivatives of selected ARCTIC sequences. ConTrack applies additional processing and quality improvements to produce robot joint trajectories, object tracks, contact annotations, and task metadata, including kinematic retargeting to xArm7 with XHAND1 and RB-Y1A with Sharpa.

These derivatives are provided solely for non-commercial scientific research. The original ARCTIC license governs all use of the source data and these derivatives. ConTrack grants no additional rights to the ARCTIC data.

## GRAB

Sources

https://grab.is.tue.mpg.de/

https://github.com/otaheri/GRAB

License

Software Copyright License for non-commercial scientific research purposes

The files matching `data/xhand/grab-*.h5` and `data/sharpa/grab-*.h5` are unofficial derivatives of selected GRAB sequences. ConTrack applies additional processing and quality improvements to produce robot joint trajectories, object tracks, contact annotations, and task metadata, including kinematic retargeting to xArm7 with XHAND1 and RB-Y1A with Sharpa.

These derivatives are provided solely for non-commercial scientific research. The original GRAB license governs all use of the source data and these derivatives. ConTrack grants no additional rights to the GRAB data.

## Dataset Derivative Disclaimer

ConTrack processing and retargeting do not transfer ownership of the original recordings, human-subject data, models, meshes, or annotations. Authorized recipients must comply with the original dataset licenses, access conditions, subject restrictions, citation requirements, and withdrawal requests.

## This Supermimic-dex release

ConTrack source is preserved from `f5efe18e4b1f986566b252265c4bf63f3b67c6c6` plus the winning BPTT extension `a232096726ae6c9a6523aabe3b8cd7a156bd66a0`. Copyright 2026 EmptyBlueBox and the original source-file copyright holders are retained. RSL RL and Isaac Lab license texts are included under `licenses/`.

Dex-Supermimic (`https://github.com/CumulusAlpha/Dex-Supermimic`, revision `2becb7380fed86171056f9bf4338b2a6a0c14b01`) is the training-method reference. The BPTT extension is independently implemented; this repository does not redistribute the donor repository.

The upstream asset/data sections describe the resources fetched from ConTrack. This repository does not bundle or relicense robot meshes, URDFs, ARCTIC/GRAB/DexterHand clips or generated USD. The resource downloader retains pinned provenance; access and use remain subject to the original owners' terms. The included Hammer policy was trained using the GRAB-derived reference, and its availability grants no additional rights to that reference data.
