# Generic robot-hand interface

Stage 4 defines the target-hand kinematics contract used by later retargeting stages. It is
deliberately independent of Isaac Gym, MuJoCo, ROS, and any optimization backend.

## Specification and loading

`RobotHandSpec` is a serializable, hashable YAML schema containing the robot name/version, side,
asset ID, URDF-relative path, `palm`-style base link, declared `dof_order`, neutral position,
expected topology, semantic layout, anchor profile, visual/collision policy, provenance,
assumptions, and notes. It never stores a machine-local asset root. The configuration hash is
recorded by `inspect`, `validate`, and FK reports.

`RobotHandRegistry` discovers `configs/robots/*.yaml`. `robots list` reads only these YAML files
and checks the expected URDF/manifest paths; it does not parse URDFs or meshes. `inspect`,
`validate`, FK, anchor, Jacobian, and visualization commands load the selected asset.

Asset-root precedence remains the Stage 0 policy: explicit CLI option, `ARTIMANO_ASSET_ROOT`,
`.local/config.yaml`, then `.local/assets/artimano`. Tracked configs contain only relative paths
and upstream-relative provenance.

## `RobotHandModel` API

The model exposes:

```text
name, side, base_link, dof_names, num_dofs
joint_lower, joint_upper, neutral_q
link_names, joint_names, spec, asset_manifest

forward_kinematics_base(qpos)
forward_kinematics_scene(qpos, base_pose_scene)
link_transform_base(qpos, link_name)
link_transform_scene(qpos, base_pose_scene, link_name)
keypoints_base(qpos, layout="mediapipe21")
keypoints_scene(qpos, base_pose_scene, layout="mediapipe21")
keypoint_set_base(qpos), keypoint_set_scene(qpos, base_pose_scene)
keypoint_jacobian_qpos(qpos, layout="mediapipe21")
visual_geometry_instances(qpos, base_pose_scene=None)
collision_geometry_instances(qpos, base_pose_scene=None)
qpos_from_named_dict(values), qpos_to_named_dict(qpos)
validate(), describe()
```

`qpos` accepts `[N]` or `[..., N]`. FK, keypoints, scene transforms, and Jacobians preserve
Torch autograd and support float32/float64 on CPU/CUDA. FK returns one homogeneous transform per
named link, including fixed links. Geometry inspection currently accepts one qpos because it
expands renderable instances rather than a batched mesh scene.

The primary FK backend is explicit Torch code. `forward_kinematics_reference` is an independent
NumPy implementation used by validation; it has the same URDF equations but a separate backend.
The Jacobian differentiates only finger DoFs. It has shape `[..., 21, 3, N]` and contains no base
pose derivatives.

## URDF contract

The parser supports `fixed`, `revolute`, `continuous`, and `prismatic` joints, standard limits,
mesh/sphere/box/cylinder geometry, geometry origins, mesh scale, and relative mesh paths. It
normalizes axes and rejects zero axes, duplicate names, missing links, cycles, multiple roots,
disconnected trees, unsupported joint types, and mimic joints.

For every joint the implementation uses the standard convention:

```text
T_parent_child(q) = T_parent_joint_origin @ T_joint_motion(q)
R_rpy = Rz(yaw) @ Ry(pitch) @ Rx(roll)
R_motion = exp([axis]x q)
t_motion = axis * q                 # prismatic
```

The XML order of joints does not define the public qpos order. A spec explicitly lists every DoF,
and the model maps that order to the parsed URDF joints.

## Base frame and anchors

Stage 4 defines the URDF root link as the robot base. For Arti-MANO this is `palm` and the FK
outputs are `T^B_L`. An external `T^S_B` is applied as `T^S_L = T^S_B T^B_L`. This engineering
definition does not resolve the paper's unpublished wrist-centered hand-frame orientation
(`A_ROBOT_HAND_FRAME_001`).

The robot target reuses the canonical Stage 3 `mediapipe21` layout object; it does not define a
second point order or skeleton. Anchor profiles support `link_origin`, `joint_origin`, and
`link_local_point`. A joint-origin anchor is the joint child-frame origin after ancestor
transforms and before that joint's own motion, so a revolute joint center does not move under its
own rotation. Profiles include provenance, assumptions, and a stable profile hash.
`keypoint_set_base/scene` additionally wrap the differentiable tensor with metadata containing the
layout name, coordinate source, robot/side, profile ID/version/hash, URDF hash, and asset-manifest
hash. The tensor-returning methods remain available for direct differentiable use.

Stage 7 consumes `RobotHandModel.keypoints_base()` as differentiable
MediaPipe-21 anchors, its 22 raw-radian `dof_order`, and its URDF lower/upper
bounds. The canonical robot wrist frame is derived from those anchors with the
same explicit frame profile as the source; it is not assumed to equal the URDF
`palm` axes. The resulting base seed is produced after qpos optimization and is
stored separately in the warm-start artifact.

## Geometry and extension

Visual and collision geometry are separate instance lists. Each instance records the link, kind,
geometry type/parameters, local origin, link/base/world transforms, source path, and source hash.
Missing collision geometry is reported; visual geometry is never silently substituted. Stage 4
does not sample surfaces, build SDFs, calculate collision distances, or modify external meshes.

To add another hand, add a YAML `RobotHandSpec`, an anchor profile with the canonical layout, and
an asset resolver entry if needed. The generic parser and FK code must not gain hand-specific link
names. A synthetic YAML-only hand and fixture exercise this boundary in the public tests.

No MANO-to-robot qpos conversion, bone-direction initialization, loss, inverse kinematics, or
optimization is part of this interface.
