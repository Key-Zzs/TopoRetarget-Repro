# Stage 16-C.5 PhysX oracle gate

## Current verdict

`STAGE16C5_CONTACT_ORACLE_BLOCKED`.  Stage 16-C.5A-R3 completed the required
contact-topology diagnosis and implemented the frozen robust-oracle contract,
but its independent C5C qualification fails the unchanged physical task gates
for both frozen selected traces.  Therefore C5B is not authorized, and no
Oracle optimization episode, CEM run, PPO sample, or checkpoint exists.

This supersedes the R1-only wording while retaining its evidence.  The result
is neither a harness-metric failure nor permission to change PhysX solver
settings, tolerances, reference timing, controller, effort limits, mass, or
friction.

## R3 topology diagnosis

The frozen input is factor-8 reference timing: 321 samples at 20 Hz control,
120 Hz physics, and decimation 6; the 26-D action, 764-D observation,
`finite_virtual_6d_wrist_actuator_v1`, assets, reward, termination, and
R2-G0 physics contract are unchanged.  Every topology cell runs 20 trials.

| Phase | Contact topology | Raw and derived state gate |
| --- | --- | --- |
| T0 | 1 environment, 1 simultaneous contact | PASS |
| T1 | 33 environments, 1 active contact and 32 no-contact dummies | PASS |
| T2 | 33 environments, all simultaneous contact | FAIL |
| T3 | 33 environments, staggered candidate start only | PASS, diagnostic only |
| T4 | 33 candidate contacts: 1x33, 2x16/17, 4x8/8/8/9 | all FAIL |
| T5 | 96 candidate contacts: 1x96, 4x24, 8x12 | all FAIL |

T0 passes, but no supplied natural-contact shard passes; both the raw-state
and derived-state gates fail under T2, T4, and T5.  The only valid
classification is `TRUE_CONTACT_SOLVER_NONDETERMINISM`.  T3 changes only
candidate start scheduling and is not a valid replacement for simultaneous
candidate evaluation.  In particular, sharding as small as 8x12 has not
restored a deterministic physical candidate pool.

## Robust statistical contract

R3 introduces a pure, testable `RobustOracleContractV1` for a future
independent evaluator.  It evaluates a selected action trace from frame zero
in a fresh one-environment rollout per replica; it does not restore candidate
state, transfer cross-shard state, write the object/wrist root state, apply a
hidden force, teleport, or change the reference/action trace.

- Replicas: 1, 4, or 8 for a candidate; default 4 and upgrade 8.
- Cost: `mean + 1.0 * population_std` (with a frozen 0.8 upper-CVaR also
  reported).
- Selector order: failure probability, 0.8-CVaR formal-gate violation, worst
  normalized gate margin, mean object error, mean rotation error, contact
  stability penalty, action smoothness, effort, then candidate ID.
- C5C: exactly 20 independently reset frame-zero replicas per selected trace.
  It must retain position <= 0.02 m, rotation <= 10 deg, axis <= 0.03 m,
  success >= 90%, and final reach >= 90%.

Robust statistics rank valid independent executions; they never waive a
physical task failure or turn a serial throughput result into evidence that
simultaneous contact batching is safe.

## Independent C5C result

The existing selected MuJoCo action traces were replayed unchanged only to
qualify their Isaac physical outcome.  Both reports record zero formal object
and wrist execution-state writes, no hidden force/teleport, and no reference
or action modification.

| Clip | 20-replica result | Dominant failure |
| --- | --- | --- |
| `hocap_170105` | 0% success, 0% final reach; position 0.00827 m, axis 0.04665 m, rotation 47.48 deg | `FAILURE_OBJECT_ORIENTATION` |
| `hocap_170650` | 0% success, 0% final reach; mean position 0.03841 m, axis 0.05492 m, rotation 26.66 deg | `FAILURE_OBJECT_AXIS_POINT` |

Neither trace meets the unchanged C5C gates, so a robust contract is
implemented but not physically qualified.  It does not authorize C5B or C.6.

## Scope stop and future authority

C5B, C.6/PPO, C.7, checkpoints, and real-robot work remain out of scope.  A
future goal needs explicit authority to propose and freeze a new physical
repair; it cannot silently relax a gate or treat the serial robust runner as a
solution to the contact-topology failure.  Historical MuJoCo Oracle evidence
does not authorize an Isaac Lab Oracle or PPO.

The authoritative machine-readable reports are intentionally ignored local
evidence:

- `.local/reports/stage16c5a_r3_contact_topology_final/contact_topology_diagnosis.json`
- `.local/reports/stage16c5a_r3_robust_oracle_final_retry1/robust_oracle_report.json`
