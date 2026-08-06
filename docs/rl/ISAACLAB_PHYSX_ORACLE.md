# Stage 16-C.5 PhysX oracle gate

## Current verdict

`STAGE16C5_PHYSX_ROBUST_ORACLE_PARTIAL`. Stage 16-C.5A-R4 implements the
distributional contract and persistent GPU pool, and C5B implements an actual
three-iteration multi-horizon robust CEM. The frozen distribution gate fails
for every contact-bearing phase on both clips, and both bounded B2 rollouts
finish with formal failure probability 1.0. B3 and the new formal C5C are
therefore `NOT_STARTED_GATE_BLOCKED`. C.6/PPO is not authorized: started is
false, samples are 0, and checkpoints are 0.

This supersedes the R1-only wording while retaining its evidence.  The result
is neither a harness-metric failure nor permission to change PhysX solver
settings, tolerances, reference timing, controller, effort limits, mass,
friction, reward, termination, or physical gates.

## R4 distributional replication result

`DistributionalReplicationContractV1` is frozen before candidate inspection.
For 20 natural and 20 candidate replicas it covers object pose/velocity, wrist
state, finger state, tracked links, all reward components, contact count/force/
impulse, termination reason, and success probability. The immutable metrics
are mean, variance, p95, per-feature Wasserstein, RBF MMD, total-variation
termination divergence, and 95% Wilson success intervals. Every limit is twice
the p95 deterministic half-split envelope measured from the natural population
only; no result-dependent metric or gate adjustment is permitted.

Both clips pass pre-contact. `hocap_170105` fails all three contact-bearing
phases in state/task fields; `hocap_170650` fails contact onset in reward
components and fails sustained/post-contact state/task fields. Termination
divergence remains zero, so this is not a reason-code or auto-reset artifact.
The required historical conclusion `SAME_SCENE_CONTACT_DIVERGENCE` /
`TRUE_CONTACT_SOLVER_NONDETERMINISM` is retained unchanged.

## Persistent pool and robust CEM

One persistent DirectRLEnv is reused per benchmark/CEM process. All required
layouts pass unique allocation, deterministic per-iteration slot permutation,
ranking invariance under two mappings, and zero hidden rollout writes.

| Population x horizons x replicas | Candidate envs | VRAM | Vector control steps/s | Dispatch | Aggregation |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 x 3 x 4 | 384 | 2351 MiB | 2.529 | 6.24 ms | 1.56 ms |
| 48 x 3 x 4 | 576 | 2429 MiB | 1.716 | 6.26 ms | 2.33 ms |
| 32 x 3 x 8 | 768 | 2507 MiB | 1.315 | 6.48 ms | 2.02 ms |

The selection rule first requires CEM compatibility, then chooses throughput;
32 x 3 x 4 is selected. `RobustMultiHorizonCEMV1` uses H=[1,5,10], population
32, three iterations, eight elites, four replicas, initial std 0.35, floor
0.05, and no scored padding. The selector is strictly ordered by failure
probability, 0.8-CVaR gate violation, worst normalized margin, p95 axis,
p95 position, p95 rotation, mean tracking, contact stability, smoothness, and
effort. Candidate simulations use audited setup rewinds; selected actions use
the normal controller/PhysX path. Formal C5C, if authorized later, is explicitly
restore-free.

B1 covers pre-contact/contact/post-contact for both clips. B2 runs 30 steps per
clip and exercises both H=1 and H=10 selection. Its terminal evidence is:

| Clip | H selection | failure probability | p95 position | p95 rotation | p95 axis |
| --- | --- | ---: | ---: | ---: | ---: |
| `hocap_170105` | H1: 19, H10: 11 | 1.0 | 0.2505 m | 138.49 deg | 0.2987 m |
| `hocap_170650` | H1: 26, H10: 4 | 1.0 | 0.2248 m | 137.45 deg | 0.2887 m |

These failures are retained. B3, optimized 320-action traces, and C5C's 20
fresh episodes per clip do not exist because their prerequisite gates fail.

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

Neither historical trace meets the unchanged C5C gates. This remains R3
evidence; R4 later implements bounded C5B but also fails and does not authorize
C.6.

## Scope stop and future authority

B3, formal C5C, C.6/PPO, C.7, checkpoints, and real-robot work remain
gate-blocked. Factor-8 changes the reference's time semantics; the virtual
wrist remains an abstract actuator rather than a real robot arm. Nothing here
is a sim-to-real claim. A future goal cannot silently relax a gate, change a
frozen metric after observing results, or treat B0--B2 execution as formal
qualification.

The authoritative machine-readable reports are intentionally ignored local
evidence:

- `.local/reports/stage16c5a_r3_contact_topology_final/contact_topology_diagnosis.json`
- `.local/reports/stage16c5a_r3_robust_oracle_final_retry1/robust_oracle_report.json`
- `.local/reports/stage16c5_r4_distributional_final/`
- `.local/reports/stage16c5_r4_cem_final/`
