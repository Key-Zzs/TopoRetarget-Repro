# Stage 16-D Strict Per-Finger V4

## Status and question

Strict Per-Finger V4 is the current single-variable Stage 16-D causal PPO
experiment. It tests whether replacing V3's proximity/aggregate contact term
with source-confirmed independent-finger contact improves interaction fidelity
without degrading frozen kinematic, physics, or absolute-geometry gates.

The causal chain remains:

```text
policy action -> robot dynamics -> hand-object contact -> PhysX object dynamics
```

No external object control is a qualification requirement, not a reward bonus.

## Frozen inputs and fair initialization

The V4 input freeze hashes `SourcePerFingerContactEvidenceV1`, raw MANO/object
provenance, Reference Kinematics V2, V1 Formal20 exact pair-force telemetry,
the V3 selected contract/checkpoint lineage, formal seed manifests, and
physics/action/observation/controller contracts. Any mismatch is an input
provenance drift and blocks V4.

Each clip begins from its own V3-matched V1-L0 actor and observation normalizer
only; critic and optimizer are fresh. Cross-clip transfer and V3/V2 checkpoint
warm starts are prohibited. The actor remains 764-D and acts through the same
26-D reference residual action.

## Required evaluation protocol

Both clips train through the shared 4,194,304-sample minimum, preserving 1M,
2M, 3M, and 4M development candidates. Development selection is lexicographic:
qualified/physics success, persistent and source tip recall, lower
cross-finger compensation/flight, stability and twist, then tracking error and
the earlier checkpoint. Total reward is not a primary selection criterion.

Each selected checkpoint receives unseen Formal20 evaluation. The comparison
preserves V1, V3, and V4 results; source-contact interaction success remains a
separate qualification dimension rather than silently redefining Evaluation
Suite V2 success. A per-clip 4M effectiveness gate can authorize, but never
require, a bounded 8M/12M/16M continuation.

## Route after V4

If V4 interaction fidelity and causal-physics guards validate on both clips,
freeze the causal contact contract and continue only with Contact-ready RSI V2,
support feasibility, then gravity and friction curriculum. External guidance
or data-H2R is considered only if this causal path is insufficient.
