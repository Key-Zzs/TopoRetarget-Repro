# Stage 16-C.5A state-replication gate

## Current closeout

The current R4 state-replication gate is `REPLICATION_DISTRIBUTION_FAIL`.
Deterministic R1--R3 evidence remains immutable, including
`TRUE_FROZEN_PHYSX_BASELINE_NONDETERMINISM`, `SAME_SCENE_CONTACT_DIVERGENCE`,
and `TRUE_CONTACT_SOLVER_NONDETERMINISM`. R4 no longer asks for elementwise
same-action equality; it freezes a natural distribution, then tests whether a
candidate-state population remains inside its statistical envelope. Both
clips pass pre-contact and fail every contact-bearing phase. This is a
fail-closed replication result, not PPO authorization.

The frozen input is the C3R5 factor-8 derived reference: both original source
NPZ hashes and all 41 source keys remain unchanged, while the runtime has 321
samples at 20 Hz. The task remains 26 DoF/action, 764 observation values, and
120 Hz physics with decimation 6. `frozen_inputs.json` in the ignored C.5A
report bundle records exact source/config/asset hashes and the archive path.

## Candidate-state contract

`Stage16C5CandidateStateV1` explicitly captures per-environment simulation
state (robot joint/root and both object roots), task/reference indices, action
history, controller targets and residuals, saturation/termination buffers, and
environment origins. Capture validates field existence, shape, discrete dtype,
device, and finite floating values. Restore rebases world positions by the
destination environment origin and writes only candidate IDs during candidate
setup. Formal execution-rollout direct wrist/object state writes are prohibited
and separately audited.

Isaac Lab provides no supported API to restore PhysX solver warm-start,
contact-manifold, friction-patch, or internal constraint caches. These omitted
states are named explicitly rather than treated as cloneable.

## R4 distributional contract

`NaturalPhysicsDistributionV1` uses 20 same-initial-state, same-reference,
same-action replicas without a snapshot. It records the four frozen phases and
mean/std/variance/p95 summaries. `DistributionalCandidateReplicatorV1` then
compares 20 replicated candidates to that baseline using mean, variance, p95,
Wasserstein, MMD, termination total variation, and 95% Wilson success
intervals. Thresholds are frozen from the natural population only as twice its
p95 deterministic half-split envelope; candidate results cannot tune them.

The measured R4 outcome is:

| Clip | Pre-contact | Contact onset | Sustained contact | Post-contact |
| --- | --- | --- | --- | --- |
| `hocap_170105` | PASS | FAIL | FAIL | FAIL |
| `hocap_170650` | PASS | FAIL | FAIL | FAIL |

Termination divergence is zero throughout. The failures occur in restored
state/task distributions, especially object pose/velocity, finger/wrist
state, tracked links, and reward components. The named inaccessible PhysX
contact/solver caches remain inaccessible; they are not silently synthesized.

## R4 persistent pool

`PersistentRobustCandidatePoolV1` maintains one live GPU environment for a
whole benchmark or CEM run. Exact tested candidate capacities are 384
(32x3x4), 576 (48x3x4), and 768 (32x3x8). Each CEM iteration applies a
deterministic seed-fixed slot permutation and records candidate ID to env ID.
Two different mappings reconstruct identical logical rankings. The fastest
compatible measured layout is 384 candidates; all three layouts have zero
formal execution-rollout writes.

## O0 candidate-pool evidence

Separate CUDA processes validated 1, 32, 96, and 144 candidates with a distinct
execution environment. Each accepted run has unique environment origins,
finite state tensors, preserved clip/reference index, subset-reset isolation,
candidate-only setup writes, and zero formal execution-rollout writes. The
configured future-only schedule is 96 = three horizons x 32, with a 144 = three
horizons x 48 upgrade. It is allocation only: it implements neither CEM nor
candidate scoring.

The first 1-candidate smoke marked a nonexistent peer as contaminated. The
original partial artifact remains preserved. A bounded bookkeeping repair
reran that case without treating absence of a peer as a peer comparison; the
repair and both artifacts are recorded in `failure_transitions.jsonl`.

## Baseline and stop rule

R1 first repairs the reset boundary and raw-step harness, then runs 20 trials
for each of pre-contact, contact-onset, sustained-contact, and post-contact on
both clips (160 samples per metric). Exact same-process one-environment replay
passes; environment-origin subtraction is valid; 20 independent one-environment
and 33-environment cross-process controls are fingerprint-identical; contact
telemetry does not alter physical fingerprints. In the same-process 33-env
baseline, however, peers diverge after contact. Its frozen rule is
`max(fixed_floor, 10 * baseline_p99)` with hard caps. The measured global
tolerance exceeds caps for object position/orientation, joint position,
linear/angular velocity, and reward. The exact values are immutable in
`replication_noise_floor.json`.

Therefore O1 tensor-clone qualification, `deterministic_history_replay_v1`,
candidate independence beyond O0, and resource benchmarking are **not run**.
The fallback is permitted only after a passing baseline plus a tensor-clone
contact mismatch; it cannot be used to evade this failure. No tolerance is
softened.

## Gate boundary

C5B implementation and bounded B0--B2 evidence now exist, but B3 and formal
C5C remain gate-blocked by this R4 failure and the two B2 physical failures.
C6/PPO remains not authorized: started=false, samples=0, checkpoints=0. A
future run must first pass the same frozen distributional contract; it cannot
change metrics after seeing candidate results or relax any physical task gate.

Machine-local evidence is under `.local/reports/stage16c5a_repair_c5c_oracle/`
and `.local/reports/stage16c5_r4_distributional_final/`; it is intentionally
ignored by Git. Factor-8 changes time semantics, the virtual wrist is not a
real arm, and no result here supports PPO, checkpoints, or sim-to-real.
