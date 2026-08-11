# Stage 16-D Phase 1 — Terminal Dynamics Attribution

Phase 1 freezes the selected PPO checkpoints, their R7 frame-zero traces, the
factor-8 references, the physics asset manifests, and the action/observation/
RSI contracts before diagnosis. It never retrains a policy or changes the
nominal rollout contract.

The method audits reference terminal linear/angular twist against pose finite
differences, actual/residual terminal twist, contact timing and impulse,
support provenance, and RSI reset implementation. A gravity diagnostic must be
reported separately from nominal zero-gravity replay; it may never be called a
qualification result.

The current factor-8 reference audit finds that its stored object twist does
not agree with finite differences of its stored object poses. Therefore its
twist is not a valid object-velocity reward target. Stored reference terminal
motion is also nonzero. Until the reference retiming/twist contract is repaired
and revalidated, terminal drift cannot be attributed solely to PPO residual
twist and Phase 3 object-twist reward is not authorized.

Support provenance remains unresolved in the source/reference metadata. The
current simulator has no ground or support and explicitly uses zero gravity,
so it cannot establish source support semantics.

## Bounded physical diagnostics

`RSIStateQualityAuditV1` sampled every retained stratified state (62 states
for `hocap_170105`, 63 for `hocap_170650`), with four zero-residual replicas
and twenty control steps per state. The corresponding gravity-only diagnostic
uses the same state set and changes only gravity. It is explicitly not a
nominal qualification: gravity creates pre-contact drift risk in 68/248
`hocap_170105` rows and 75/252 `hocap_170650` rows under the declared 5 mm
criterion. The nominal zero-gravity run has no classified
`PRE_CONTACT_UNSUPPORTED` rows for either clip.

The bounded counterfactual uses frozen R7 actions and two representative formal
episodes per clip (one stable-ish and one failed; `hocap_170105` has no formal
stable episode). In CF2, a last-contact state evolved freely for one second:
at zero gravity its already-present vertical velocity persists, while at
standard gravity the object falls about 4.93 m. CF3 repeats the test for the
reference terminal state and shows the same distinction: its stored terminal
velocity persists at zero gravity and falls under gravity. Thus zero gravity
preserves residual velocity; it is not evidence that zero gravity generated
that velocity. The gravity-only same-action CF1 is also retained as a
non-nominal safety diagnostic; its unchanged safety gates terminate the selected
replays early, so it is not presented as a replacement task trajectory.
