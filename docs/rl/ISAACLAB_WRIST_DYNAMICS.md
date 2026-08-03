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
D6 joints, so the explicitly permitted finite virtual 3P+3R fallback is used.

The fixed C.3 wrist gate is at most 2 cm / 10 degrees maximum error, 1 cm / 5
degrees RMSE, and 5% force/torque saturation. All three frozen finite virtual
profiles fail both 41-key clips: conservative reaches 3.91 cm/29.45 degrees
and 4.63 cm/21.04 degrees; nominal 3.23 cm/38.34 degrees and 4.54 cm/37.10
degrees; high authority 4.10 cm/53.63 degrees and 6.81 cm/54.38 degrees.
The finite disturbance remains physical and finite, while removing virtual
authority worsens the combined position RMSE from 0.03623 m to 0.47282 m.

The result is `C3_WRIST_ACTUATION_ARCHITECTURE_BLOCKED`. No profile is active;
there is no C.2 active-profile regression. C3-1--C3-5, contact-momentum
causality, C.4, C.5, and PPO are fail-closed/not run. The non-contact wrist
gate uses live PhysX evolution and bounded force/torque at `r_wrist`, with no
rollout wrist pose/velocity or object state write; immutable task-object
termination is intentionally not evaluated in that gate.

Machine-local evidence is under `.local/reports/stage16c3r2_c5/`.
