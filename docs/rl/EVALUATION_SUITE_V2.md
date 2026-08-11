# Evaluation Suite V2

`TopoRetargetEvaluationSuiteV2` is the additive shared evaluation contract for
Stage 16-D PPO, future Multi-Clip PPO, physical curricula, and each supported
adapter. It does not remove or redefine legacy task/contact/geometry metrics.

## Coordinate and aggregation contract

Primary metrics use the common world/env frame after environment-origin removal.
Each reports a trajectory mean over observed valid frames; p95-over-time, max,
and terminal values are diagnostic. An incomplete episode reports diagnostic
metrics over observed frames but always has `kinematic_success=false`.

## Primary metrics

- `E_r`: raw object-orientation SO(3) geodesic mean, in degrees.
- `E_t`: object-root-origin Euclidean mean, in centimetres.
- `E_j`: mean Euclidean error between actual Wuji keypoints and shared
  retargeted-Wuji reference keypoints, in centimetres.
- `E_ft`: mean Euclidean error across thumb/index/middle/ring/pinky landmarks,
  in centimetres.

`EvaluationJointSetV1` maps the wrist and canonical proximal/middle/distal
link names. `EvaluationFingertipSetV1` maps each distal link root as a shared
engineering fingertip landmark; this approximation is explicit and shared by
every clip.

## Success contract

`SR_kinematic` requires strict trajectory means:

```text
E_r < 30 deg AND E_t < 3 cm AND E_j < 8 cm AND E_ft < 6 cm
```

`SR_physics` requires terminal contact/stability, contact causality,
inter-finger and absolute hand-object penetration safety, action bounds, no
hidden force, no object rollout state write, and no wrist-root teleport.
Source-relative geometry fidelity remains a separate legacy diagnostic.

`SR_qualified = SR_kinematic AND SR_physics`. Future bimanual trajectories
require the object metrics and the joint/fingertip pass for both hands.

## Reference-contact evaluation V1

Reward V3 adds diagnostic contact behaviour without redefining any success
gate: reference expected-contact fraction, actual fingertip--active-object
contact fraction, expected-contact recall, per-finger recall, unexpected-
contact rate, persistent-contact recall, longest loss gap, loss/recontact
counts, terminal contact/expected-contact, and force mean/p95/max plus total
impulse. A persistent reference-contact window is any run of at least three
control steps with one or more active reference mask entries. Actual contact
uses the same pair-specific filtered PhysX force source as the reward and its
pre-frozen numerical floor.

## Reference-kinematics V2 traces

Phase 3 traces carry `reference_kinematics_version=2`, signed world-frame
actual and reference object twists, their residual norms, and the two frozen
Reward V2 components.  These fields are additional diagnostics: they do not
replace `E_r`, `E_t`, `E_j`, `E_ft`, or any physics gate.  A terminal reference
that is still moving must be reported as a terminal-semantics mismatch rather
than silently changing the absolute terminal-stability definition.

Reward V3 traces additionally carry the reference mask, actual five-fingertip
mask, exact fingertip-object pair force, force magnitude/scale, and
`r_contact`. These are diagnostics and replay fields; they do not grant an
episode physics qualification by themselves.
