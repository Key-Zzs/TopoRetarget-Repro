# Stage 4 — Generic robot-hand kinematics interface and Arti-MANO target adapter

## Objective and scope

Stage 4 implements the target-hand function `P^r(q)` for a generic URDF hand interface and real
Arti-MANO RH/LH adapters. It accepts named finger-joint positions and an external homogeneous
base pose. It does not solve for q, convert MANO/MediaPipe to qpos, or implement any paper loss.

## Implementation

| Area | Implementation |
| --- | --- |
| schema/model | `src/toporetarget/robots/spec.py`, `base.py` |
| contract/registry/Arti-MANO | `contracts.py`, `registry.py`, `paths/assets.py`, YAML configs |
| URDF | `robots/urdf/parser.py`, `model.py`, `kinematics.py` |
| geometry | `robots/urdf/geometry.py`, `robots/visualization.py` |
| anchors/reports | `anchors.py`, `reports.py` |
| CLI | `src/toporetarget/cli/robots.py` and `cli/main.py` |
| public tests | `tests/fixtures/synthetic_hand.urdf`, `tests/unit/test_robot_*`, `tests/local_assets/test_artimano_stage4.py` |
| artifact generator | `scripts/generate_stage4_artifacts.py` |

The primary FK backend is differentiable Torch. A separate NumPy reference FK cross-checks every
link transform. URDF RPY uses `Rz(yaw) Ry(pitch) Rx(roll)`; fixed links remain explicit. qpos order
is declared in YAML rather than inherited from XML order.

## Asset evidence

The tracked vendor snapshot is imported from ManipTrans commit
`a3d08cfe3c3a5868a7f057533bcaf759c5af4705`. The manifest SHA-256 is
`c9601ed490bcec6f6d672d1ae4d8fd3f08724e357bf977cf63553a94cbdc3cf2`; RH/LH URDF hashes are
`21800ccf73b980ac7927b97d12921ce5498ac5c1579bd65e84d269a01ef5b660` and
`472b84cfb2197ef7f818a66b62539b093f3fd5c93fd0fa8e5f8931d689eccd29`. The import contains 98
files, including 96 meshes, and has zero unresolved references. The tracked URDFs only rebase mesh
filenames from the upstream flat layout; `asset_comparison.json` and `numerical_regression.json`
record exact payload, topology, FK, anchor, Jacobian, and mesh-transform equality against the
legacy tree. `.local/assets` remains ignored and untracked for compatibility only.

## RH/LH validation

Both actual URDFs independently report 28 links, 27 total joints, 22 actuated joints, five fixed
joints, and base `palm`. Both use the 22-name order in `ARTIMANO_ADAPTER.md`, all-zero neutral q,
21 anchors, 21 visual instances, and 16 collision instances. The candidate anchor mapping was
confirmed unchanged. The shared profile hash is
`872900ba7c252562d0d84de7f75722d25b0026be238bc8e8af0cf088a909b04e`.

Float64 validation over neutral plus three deterministic random poses (`seed=4`) passed for both:

| Side | FK translation max | FK rotation geodesic max | base equivariance max |
| --- | ---: | ---: | ---: |
| RH | `2.7755575615628914e-17` m | `1.2749440250986722e-16` rad | `0.0` m |
| LH | `2.7755575615628914e-17` m | `1.2749440250986722e-16` rad | `0.0` m |

The float64 central-difference Jacobian uses `epsilon=1e-6`:

| Side | Shape | Max absolute | RMSE | Relative | Result |
| --- | --- | ---: | ---: | ---: | --- |
| RH | `[21, 3, 22]` | `1.1891567591737484e-11` | `1.0901791404256092e-12` | `2.262714149440586e-10` | pass |
| LH | `[21, 3, 22]` | `1.1891567591737484e-11` | `1.0901791404256092e-12` | `2.262714149440586e-10` | pass |

The reports also confirm finite anchors, no zero-length skeleton bone, named limits, and separate
visual/collision loading. Fixed fingertips have visual-only spheres; no collision replacement is
performed.

## Generated artifacts and tests

The required local outputs are in `.local/reports/stage4/`: RH/LH inspect JSON, validation JSON/CSV,
anchor CSV, Jacobian JSON, neutral visual/collision PNGs, random both-geometry overlays,
`artimano_both_neutral.png`, and `asset_integrity.json`.

Public synthetic tests and opt-in local tests pass. The paper fidelity checker remains green with
`stage4_robot_keypoint_forward_kinematics` marked `implemented_with_assumptions`. README workflow and
TODO status, both development logs, assumptions, and open questions were updated. No retargeting
optimization or Stage 5 work is included.

## Assumptions and blockers

- `A_ROBOT_KEYPOINT_ANCHORS_001`: the paper does not publish target-hand anchor selection; profiles are explicit URDF anchors.
- `A_ARTIMANO_KEYPOINT_MAPPING_001`: thumb semantics and multi-axis coincident joint handling are engineering mappings.
- `A_ROBOT_BASE_FRAME_001`: this project uses URDF root `palm` as base; the paper's wrist-frame orientation remains unresolved.
- `A_ARTIMANO_COLLISION_COVERAGE_001`: fixed tip spheres are visual-only and distal collision meshes are reported, not synthesized.
- `A_ARTIMANO_DOF_ORDER_001`: order is explicit and audited against both URDFs and ManipTrans.
- `A_ROBOT_HAND_FRAME_001`: the paper's exact robot wrist-centered frame remains pending author confirmation.

## Definition of done

The generic schema/parser/FK/reference backend, RH/LH configs, anchors, Jacobian, geometry,
registry/CLI, tracked asset provenance, synthetic tests, local validation, artifacts, and
synchronized docs are complete.
Stage 4 intentionally ends before retargeting; Stage 5 remains not started.
