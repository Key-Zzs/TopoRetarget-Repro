# OakInk2 O5R-D2A RetargetObjectiveV2 Design

## Decision

```text
RETARGET_OBJECTIVE_V2_DESIGN_STATUS=NO_CANDIDATE_READY
SELECTED_OBJECTIVE=NONE
RETARGET_OBJECTIVE_V2_SHA256=null
```

O5R-D2A repaired the production-context binding used for objective development and compared two interaction-first formulations. It did not freeze an ObjectiveV2 because neither candidate completed all development windows while satisfying the frozen non-regression contract. No independent validation or downstream stage is authorized.

## Context binding result

`ProductionObjectiveContextBindingV2` explicitly binds the active/current source frame, previous source/runtime frame, previous robot state, continuous predicted base and finger state, base-correction reference, object pose, robot/side/DOF mapping, and source hand. The frame-zero binding preserves V1's historical no-temporal branch.

On all 55 consumed Q0 states, the V2 binding plus the unchanged V1 formula reproduced every objective component and total exactly:

```text
PRODUCTION_OBJECTIVE_CONTEXT_BINDING_V2=PASS
N=55
max total absolute difference=0.0
max component absolute difference=0.0
```

## Candidate definitions

Candidate A minimizes per-frame `E_IM`, then minimizes the V1 fidelity terms excluding interaction while retaining the primary result. It was rejected because it continues applying interaction pressure below the Semantic V1 target and had weaker technical completion.

Candidate B minimizes the dimensionless hinge

\[
\max(E_{IM}/10^{-4}-1,0)^2,
\]

then minimizes the V1 fidelity terms excluding interaction subject to interaction retention. Its interaction pressure is exactly zero at and below the frozen target. Candidate B is promising but not ready.

Both candidates retain the production collision constraints and Wuji joint bounds. Wrist/bone limits come from frozen Semantic V1; continuity limits come from `wuji_continuous_full_state_v1`. No threshold or budget was fitted to DEV1 Q1 outcomes, and no weight grid search was used.

## Development evidence

- All 25 B1 Q1 endpoints improve interaction and remain collision-feasible, but none satisfies the already-frozen continuous-profile correction limits. Candidate B correctly saturates on all seven LOW controls; Candidate A does not.
- In 15 deterministic single-frame cases, Candidate B completed 15/15, introduced no hard validity violation, and moved 20% of initially above-target cases to at or below target.
- Three deterministic 20-frame sequential windows added 51 new development frames. The future exclusion ledger therefore contains 106 unique DEV1 frames.
- Candidate B reduced the MID-window p95 from `1.379943803e-4` to `9.710502070e-5` and preserved the LOW window below target.
- Candidate B's HIGH window remained incomplete: only 10/20 selected states were technically usable under the continuous non-regression budget, and p95 remained `1.939272528e-4`.

The unchanged Semantic V1 evaluator passes Candidate B's MID and LOW development windows and fails its HIGH window on interaction geometry. These are development compatibility results, not trajectory validation.

## Boundary

Objective V1, Semantic V1, StructuredSolverV1, Manifest V2, Split V2, the Wuji/MANO authorities, and the saved DEV1 trajectory remain unchanged. O5R-D2A ran zero DEV1 full reruns/refinements and zero DEV2 solves. It created no SparseValidationV2 or WindowValidationV2 set and consumed no heldout/certification data.

The next valid action is another bounded objective/search-design phase addressing Candidate B's HIGH-window continuity/technical failure. Independent certification remains blocked until a candidate is frozen before untouched, zero-overlap sparse and contiguous-window sets are selected.

