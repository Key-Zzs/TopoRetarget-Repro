# Stage 16-C.5A state-replication gate

## Current closeout

The state-replication gate is
`STAGE16C5A_PHYSICS_CONTRACT_CHANGE_REQUIRED`, with reason
`TRUE_FROZEN_PHYSX_BASELINE_NONDETERMINISM`. This is a fail-closed physical
contract result, not a PhysX Oracle, CEM, policy, or PPO result. No full Oracle
episode, formal 20-episode evaluation, checkpoint, or PPO sample was created.

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

C5B requires passing C5A O0/O1 evidence. It is not authorized here. C6/PPO
also remains not authorized. A future run must first reproduce a passing natural
baseline under the same frozen inputs and hard caps, then separately qualify
tensor clone across all phases before considering the bounded fallback or C5B.

Machine-local evidence is under
`.local/reports/stage16c5a_repair_c5c_oracle/` and is intentionally ignored by
Git. Any next experiment requires separate authorization for a minimal frozen
PhysX contract change; it must not be presented as O1, C5B, C5C, or PPO work.
