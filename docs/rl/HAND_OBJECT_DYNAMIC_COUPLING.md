# Stage 16.1a hand--object dynamic coupling

This document records the bounded A--E diagnosis for the two immutable 41-frame,
20 Hz, 20 DoF HOCap references. It is an engineering-controllability report, not a
paper-policy or author-exact simulator claim.

## Frozen boundary

The original failure is frozen at
`.local/archive/stage16_controllability_failure_baseline_20260801T060846Z_189b2f8/`.
The decision evidence is
`.local/reports/stage16_dynamic_coupling_v1_rerun1/`. References, object meshes,
formal 5 cm position/axis and 45 degree orientation gates, fixed base, action
dimension, residual semantics, and PPO reward/observation are unchanged.

## A--E result

- Step A — `STEP_A_PD_PASS`: dynamic hand / kinematic object reaches all 41 frames;
  worst joint RMSE is 0.01594 rad and worst link RMSE is 0.865 mm.
- Step B — `CONTACT_FORCE_CLOSURE_INSUFFICIENT`: collision geoms and filtering are
  active at later frames, but frames 0/5/10 have no actual or expected proximity
  contact for either clip, including global 0/1/2/5% preload probes. A 5% preload
  improves later contact evidence for both clips but cannot create pre-gate support.
- Step C — C0--C3 are preserved in the report. C3 (hand reference velocity) wins the
  predeclared shared tie-break, but does not change the earliest frame-5/6 crossing.
- Step D — `ObjectAwareResidualOracle` uses cloned MuJoCo state and central finite
  differences to return only a bounded 20D finger residual. It has local rank 20,
  but cannot complete either episode; it never writes object qpos/qvel or applies
  object force.
- Step E — fixed H=5/H=10 shooting identifies local descent after contact becomes
  available, not an early full-trajectory solution. The evidence therefore classifies
  the current setup as `REFERENCE_DYNAMICAL_INFEASIBILITY`.

## Consequence

`STAGE16_1_CONTROLLABILITY_BLOCKED` and `STAGE16_2_ENTRY_NOT_AUTHORIZED` remain in
force. PPO must not be started to obscure the unsupported early object motion. A future
experiment needs separately authorized, reference-preserving evidence for early support
contact within the same fixed-base/20D protocol, or must explicitly define a different
protocol rather than silently adding a base action or direct object control.
