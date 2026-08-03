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

## C.3R2--C.5 fail-closed result

C3-0's fully kinematic frame/reference contract is validated using derived
canonical-URDF FK targets, with the frozen stored link field preserved. Path A
is exhausted before dynamic qualification: five reference-target response maps
exceed its frozen condition-number maximum of 4000. The generated six-axis
`PhysicsJoint` D6 wrapper imports, but live GPU tensor inspection exposes zero
D6 joints, so the explicitly permitted serial 3P+3R articulation fallback is
used. It contains three orthogonal prismatic joints, three revolute joints,
and the frozen Wuji hand, for 26 total tensor DoFs. The anchor and five tiny
intermediate links are an abstract engineering wrist, not a real arm. Policy
rotation remains a rotation-vector/quaternion residual; only the final SE(3)
target is converted to serial XYZ joint coordinates. The observed pitch
singularity margin stays above 78 degrees.

The fixed C.3 wrist gate is at most 2 cm / 10 degrees maximum error, 1 cm / 5
degrees RMSE, and 5% force/torque/velocity saturation. All three globally
shared profiles fail both 41-key clips. The strongest bounded profile reaches
1.13 cm/17.59 degrees and 1.09 cm/19.57 degrees maximum error, 0.64/0.55 cm
position RMSE, 7.29/7.55 degrees rotation RMSE, and 21.25%/18.75% torque
saturation. FK localization is clean (at most 0.04 degrees rotation mismatch
and about 1e-7 m translation mismatch), proving that the remaining error is
joint-drive tracking rather than a serial-rotation convention error. The
finite disturbance remains physical and finite, while removing authority
strongly worsens tracking.

The result is `C3_WRIST_ACTUATION_ARCHITECTURE_BLOCKED`. No profile is active.
The strongest candidate separately passes bounded C.2 runtime smokes at 1/128
environments, preserving all 26 action bases, 764-D observations, a 64-of-128
subset reset, and no rollout wrist/object state write; it does not select a
C.3 profile. C3-1--C3-5, contact-momentum causality, C.4, C.5, and PPO are
fail-closed/not run. The non-contact wrist gate uses live PhysX evolution and
bounded articulation drives, with no rollout wrist pose/velocity or object
state write; immutable task-object termination is
intentionally not evaluated in that gate.

Machine-local evidence is under `.local/reports/stage16c3r2_c5/`.
