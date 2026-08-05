# Stage16-C.5A-R3 Contact Topology and Robust Oracle Handoff

## Final status

`STAGE16C5_CONTACT_ORACLE_BLOCKED`.  R3 completes the requested diagnosis and
implements the robust-oracle contract, but does not physically qualify an
Isaac Lab Oracle.  The necessary C5C 20-replica gate fails for both frozen
selected action traces.  C5B, C.6/PPO, C.7, checkpoints, and real-robot work
are not authorized; PPO remains 0 samples and 0 checkpoints.

The current C3 and C4 records remain valid and unchanged: C3 uses the
factor-8-derived 321-sample reference at 20 Hz control/120 Hz physics with
decimation 6, 26-D action, 764-D observation, and
`finite_virtual_6d_wrist_actuator_v1`; C4 is the validated 4096-environment
aggregate-contact benchmark at 700.35 samples/s.  This R3 serial diagnostic
does not revise either result and does not make the virtual wrist a real arm.

## Frozen scope and implementation

R3 preserves reference/action hashes, assets, physics contract, reward,
termination, controller, effort limits, mass/friction, solver settings, and
hard caps.  It forbids direct object pose/root writes, wrist root writes,
hidden forces, and teleportation.

The implementation adds the exact T0--T5 topology matrix, pure-Python
classification and sharding contracts, a testable robust statistical selector,
and a parent/worker runner.  The runner uses one environment and a fresh
frame-zero reset for every replica; it has no candidate-state restore or
cross-shard state transfer.  Its serial form is deliberate because the
natural simultaneous-contact population is not a valid deterministic pool.

## Topology evidence

Every cell runs 20 trials with the frozen factor-8 inputs.  `raw` means the
direct-state fingerprint comparison; `derived` means the task metric
comparison.

| ID | Natural contact topology | Raw | Derived | Result |
| --- | --- | --- | --- | --- |
| T0 | 1 scene / 1 active | stable | stable | PASS |
| T1 | 33 scenes / 1 active, 32 no-contact dummies | stable | stable | PASS |
| T2 | 33 scenes / 33 simultaneous contacts | divergent | divergent | FAIL |
| T3 | 33 scenes / staggered starts | stable | stable | PASS, diagnostic only |
| T4 | 33 contacts: 1x33 | divergent | divergent | FAIL |
| T4 | 33 contacts: 2x16/17 | divergent | divergent | FAIL |
| T4 | 33 contacts: 4x8/8/8/9 | divergent | divergent | FAIL |
| T5 | 96 contacts: 1x96 | divergent | divergent | FAIL |
| T5 | 96 contacts: 4x24 | divergent | divergent | FAIL |
| T5 | 96 contacts: 8x12 | divergent | divergent | FAIL |

The classification is `TRUE_CONTACT_SOLVER_NONDETERMINISM`.  T0 passes, but
there is no passing natural-contact shard; T3 changes only candidate start
time and is not evidence for a candidate-pool repair.  This excludes both
`SINGLE_SCENE_CONTACT_BATCHING_FAILURE` and `HARNESS_METRIC_FAILURE` under the
frozen test.

## Robust contract and C5C

`RobustOracleContractV1` permits 1/4/8 independent replicas per candidate,
uses `mean_cost + 1.0 * population_std` (and reports upper CVaR at alpha 0.8),
and orders candidates lexicographically by failure probability,
CVaR formal-gate violation, worst normalized margin, mean object error, mean
rotation error, contact stability, action smoothness, effort, and candidate
ID.  C5C evaluates exactly 20 independent frame-zero replicas and retains the
unmodified gates: position <= 0.02 m, rotation <= 10 deg, axis <= 0.03 m,
success >= 90%, and final reach >= 90%.

| Clip | C5C result | Reason |
| --- | --- | --- |
| `hocap_170105` | 0/20 success and final reach; position 0.00827 m, axis 0.04665 m, rotation 47.48 deg | `FAILURE_OBJECT_ORIENTATION` |
| `hocap_170650` | 0/20 success and final reach; mean position 0.03841 m, axis 0.05492 m, rotation 26.66 deg | `FAILURE_OBJECT_AXIS_POINT` |

Both qualification reports audit `object=0` and `wrist=0` formal
execution-state writes; no hidden force/teleport or action/reference mutation
is reported.  Robust ranking therefore remains implemented but physically
unqualified.  It cannot waive those C5C failures.

## Serial resource measurement

The runner measures the required 32/96 candidate counts with 1/4/8 replicas,
but calls each rollout in a fresh one-environment Isaac worker.  The final
machine-readable report records the effective rollouts, wall time, effective
rollouts/s, GPU samples/VRAM, and IPC overhead for all six cells.  These are
safe serial-dispatch resource measurements, not a claim of simultaneous
contact-batch throughput and not interchangeable with C4's 4096-environment
benchmark.

| Candidates x replicas | Clip | Effective rollouts | Wall time (s) | Rollouts/s | VRAM peak (MiB) | IPC (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 32 x 1 | `hocap_170650` | 32 | 133.322 | 0.240021 | 4469 | 0.824 |
| 32 x 4 | `hocap_170105` | 128 | 555.057 | 0.230607 | 4472 | 0.858 |
| 32 x 8 | `hocap_170105` | 256 | 1108.492 | 0.230944 | 4723 | 0.818 |
| 96 x 1 | `hocap_170650` | 96 | 394.084 | 0.243603 | 4469 | 0.809 |
| 96 x 4 | `hocap_170105` | 384 | 1662.453 | 0.230984 | 4426 | 0.771 |
| 96 x 8 | `hocap_170105` | 768 | 3308.348 | 0.232140 | 4726 | 0.859 |

Every row reports no hidden execution-state writes.  All six use a preexisting
selected contact trace only; none is a CEM/Oracle optimization episode.

## Evidence and reproduction

The ignored local outputs are the evidence authority:

- `.local/reports/stage16c5a_r3_contact_topology_final/contact_topology_diagnosis.json`
- `.local/reports/stage16c5a_r3_robust_oracle_final_retry1/robust_oracle_report.json`

The worker reports below the same roots preserve exact per-topology and
per-replica results.  The first robust output directory was preserved after a
pre-runtime import/configuration defect; `_retry1` is the authoritative
completed retry and does not overwrite prior evidence.

Focused CPU contracts are covered by:

```bash
env PYTHONPATH=src conda run --no-capture-output -n toporetarget-isaaclab \
  pytest -q tests/rl/isaaclab/test_stage16c5_replication.py \
  tests/rl/isaaclab/test_stage16c5a_r3_topology_robust.py
```

## Next authority boundary

A later goal may propose a new physical repair only after explicitly freezing
its scope and re-qualifying it.  It may not silently loosen tolerances, alter
the controller/reference/solver, write object/wrist state, or treat robust
statistics as a substitute for a failed physical task.  Until then, C5B and
PPO remain stopped.
