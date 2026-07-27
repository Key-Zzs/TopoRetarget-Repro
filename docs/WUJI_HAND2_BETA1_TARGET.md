# Wuji Hand2 Beta1 target-hand integration

This repository tracks Wuji Hand2 Beta1 as a generic target-hand instance. The integration is
W0/W1 infrastructure and bounded validation; it is not a claim of original Wuji hardware
reproduction, deployment calibration, or paper-level zero-shot transfer.

## Asset boundary

The vendor bundle is `third_party/robot_hands/wuji_hand2_beta1/`. It contains only the approved
Hand2 Beta1 body payload: RH/LH URDFs, RH/LH MJCFs, their referenced STL meshes, and the upstream
MIT license. STEP/USD, ROS URDF variants, ROS/package files, and other upstream trees are excluded.
See [`WUJI_HAND2_ASSET_PROVENANCE.md`](WUJI_HAND2_ASSET_PROVENANCE.md).

The upstream source is `wuji-technology/wuji-description`, requested ref `release/v2026.7.23`,
resolved locally as commit `2b57d2621caed4e65207bb767ba25fc8eaec0881` through the existing remote
tracking ref. The tag with the same version string is a different commit and was not substituted.

## Registered instances

| Robot ID | Side | Root | Links | Joints | Actuated DoF | Fixed joints | URDF | MJCF |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `wuji_hand2_beta1_rh` | right | `r_wrist` | 26 | 25 | 20 | 5 | `urdf/right.urdf` | `mjcf/right.xml` |
| `wuji_hand2_beta1_lh` | left | `l_wrist` | 26 | 25 | 20 | 5 | `urdf/left.urdf` | `mjcf/left.xml` |

Both instances are loaded by `RobotHandRegistry` from YAML. The public qpos order is explicit in
`configs/robots/joint_orders/`; it is not inferred from XML order and it is not padded to the
historical Arti-MANO width. The RH/LH axes and limits remain independent source data.

## Generic integration

The integration uses the F0 `RobotHandSpec`/`RobotHandModel`/`RobotHandRegistry` contract:

- generic URDF parsing, NumPy/Torch FK, qpos bounds, anchors, geometry instances, and Jacobians;
- MediaPipe-21 anchors derived from URDF joint/link origins and official MJCF tip sites;
- separate visual/surface, URDF collision, and MJCF collision profiles;
- generic Stage 7 warm-start and Stage 8 Eq. (7) evaluation with dynamic DoF widths;
- bounded Stage 9 objective/constraint construction only.

There is no `WujiHand2Beta1Adapter`, Wuji-specific retargeting pipeline, Wuji-specific solver,
robot-name conditional in Stage 7/8/9, or hardware-control code.

## Validation entry points

```bash
PYTHONNOUSERSITE=1 python -m toporetarget robots inspect wuji_hand2_beta1_rh
PYTHONNOUSERSITE=1 python -m toporetarget robots validate wuji_hand2_beta1_rh
PYTHONNOUSERSITE=1 python -m toporetarget robots validate wuji_hand2_beta1_lh
PYTHONNOUSERSITE=1 python scripts/wuji_hand2_pipeline_smoke.py
```

The bounded smoke uses the existing airplane canonical cache and frames `[240,243)`. It validates
Stage 7 warm-start, source-only Stage 8 graph/evaluation, collision QuerySet construction, the
generic Stage 9 objective, signed-distance constraints, and the analytic constraint Jacobian. It
does not run a formal Stage 9 optimizer or create a final Wuji trajectory.

## Next boundaries

W2 is the main-branch full Wuji retargeting milestone: at least three watertight clips, full
Stage 7–9 execution, independent contact/collision audits, and rejection-safe reports. S1 is the
`develop/pene-loss` generic SDF penetration-loss branch, initially validated on Arti-MANO. I1
updates that branch to the latest main and validates Arti-MANO plus Wuji under baseline/SDF
conditions. W3 covers export; R0/R1 cover MJCF playback/PD and PPO tracking. None is claimed here.
