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
