# Isaac Lab wrist-dynamics diagnosis

## Scope and frozen boundary

This is an engineering diagnostic for C.3 only. It preserves the two immutable
41-frame, 20 Hz world-wrist references, the 26-D action and the C.2 observation
contract. It does not authorize C.4, C.5, PPO, or a physical-control claim.

## What was established

- The controller samples every 120 Hz substep: translation is cubic Hermite,
  orientation is shortest-arc SLERP, and the first/final samples equal the two
  frozen 20 Hz keys.
- Wrenches are written through the instantaneous composer on every substep, so
  their world-to-link conversion is not held at a stale wrist pose.
- A baseline-subtracted real PhysX ±x/±y/±z force/torque probe passes. Root
  quaternions are `wxyz`, root twist is world-frame, and a positive world wrench
  produces the expected signed world-frame response.
- F0 (no finger drive), F1 (zero targets), and F2 (reference targets) identify
  a coupled response. The F2 static body-frame matrix is retained in V3 for
  diagnosis, not accepted as trajectory control: it fails the 10-step run.

## C.3R3/R4 joint-dynamics closeout

C3-0's fully kinematic frame/reference contract remains validated using
derived canonical-URDF FK targets, with the frozen stored link field preserved.
The generated six-axis `PhysicsJoint` D6 wrapper imports but exposes no live GPU
tensor joint, so the explicitly permitted serial 3P+3R articulation is used.
It contains three orthogonal prismatic joints, three revolute joints, and the
frozen Wuji hand, for 26 total articulation DoFs. The fixed anchor and tiny
intermediate links are an abstract engineering wrist, not a real arm.

C3R4 corrected a physics-boundary defect without changing the frozen keys or
20 Hz timing. Six pre-step controller calls now sample boundaries 0/6 through
5/6; boundary 6/6 is observed after the sixth 1/120 s physics step and equals
key k+1. Reset initializes the explicit wrist velocity from the analytic joint
reference. The runtime dynamics path uses the full 26x26 PhysX generalized
mass matrix, keeps the wrist-finger coupling block, and evaluates live
Coriolis/centrifugal plus gravity compensation at each substep. Gravity is
zero, but the bias term is not assumed zero.

The prior bounded-MPC "worker terminated" result was false: the reporter read
`latest["gain"]` for an MPC result and raised `KeyError` after the first
interval. With exception persistence and controller-specific fields, both
workers complete all 41 frames. A six-substep trace with
`CUDA_LAUNCH_BLOCKING=1` records finite A/B, Hessian, unconstrained/projected/
applied effort and every `apply_action`/scene-write/sim-step/scene-update
boundary; its Isaac Kit log contains no CUDA or PhysX execution error.

The original V1 identification's fit R2 does not generalize: withheld normalized
RMSE is 0.06954 for one step and 0.77685 for six substeps. Raw M_ww condition
numbers span 686--1318 (180--317 after unit scaling), while the Hessian reaches
10051. The fixed projected-gradient step violates its spectral stability bound
at every audited node, so the implementation now caps the step by the inverse
largest Hessian eigenvalue without changing cost, horizon, iteration count, or
effort limit. A unit-scaled, per-substep affine V2 model obtains fit R2
0.999959, yet independent absolute holdout RMSE remains 0.09453/0.62331 for
one/six steps; both predeclared diagnostic gates fail.

Both full-articulation computed-torque profiles fail both fixed clips. The
final V2 MPC also fails: `hocap_170105` reaches 1.961 m maximum position error,
119.13 degrees rotation RMSE, and 44.58% maximum per-joint saturation;
`hocap_170650` reaches 0.777 m, 114.21 degrees, and 6.25%. Every run remains
finite, completes 41/41 frames, and performs zero rollout wrist/object state
writes, but none passes the 2 cm / 10 degree maximum, 1 cm / 5 degree RMSE, and
5% saturation gate.

The result is `C3_EXPLICIT_WRIST_FINITE_EFFORT_TRACKING_EXHAUSTED` and
`C3_WRIST_ACTUATION_ARCHITECTURE_BLOCKED`; no controller is active. Contact
causality and full C.3 are not resumed, C.4/C.5 remain gate-blocked, and PPO is
not authorized. Machine-local C3R4 evidence is under
`.local/reports/stage16c3r4_mpc_holdout_c4/`.
