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

## Fail-closed result

The C.3 wrist gate is at most 2 cm and 10 degrees. The final shared 41-step
profile reaches 3.35 cm and 23.00 degrees. Raising authority to 100 N/6 Nm is
worse (8.53 cm and 83.3% force saturation). C.3 wrist dynamics is therefore
`FAIL`; no downstream benchmark, oracle, or PPO run is authorized.

Machine-local evidence is under
`.local/reports/stage16c3_repair_c5_oracle/wrist_*.json`.
