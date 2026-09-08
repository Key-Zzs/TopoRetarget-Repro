# OakInk2 O5R-B parallel diagnostics

`ThumbFeasibilityQualification` is a sparse, diagnostic-only experiment on a
frozen production trajectory.  It holds wrist, target object, all non-thumb
DOFs, geometry, and limits fixed; only the authority-derived Wuji thumb DOFs
vary.  It reports the production thumb residual and the
`BEST_OBSERVED_FEASIBLE_THUMB_RESIDUAL` found by a deterministic bounded
search.  The latter is not a certified global lower bound.

This distinction separates an embodiment limitation from solver
suboptimality: small current-to-best-observed gaps together with limit or
kinematic evidence support an embodiment interpretation, while large observed
improvements support sparse thumb-retarget repair.  It does not change
Semantic Validity V1 or create Semantic V2.

`FirstFrameSeedAuthorityV2` is an episode-agnostic, versioned candidate set:
neutral, joint-range midpoint, lower quartile, and upper quartile.  It rejects
invalid data, records bounded probes, ranks successful probes first and then
their final objective with a fixed tie-break, and finally uses the frozen
Stage-7 solver math.  It does not change the objective, tolerance, joint
limits, solver algorithm, stage definitions, or production `max_nfev`.

Any future DEV2 run is an `O5R-C` standalone recovery production solve, not a
reconstruction of the original DEV1-to-DEV2 warm timing benchmark.  It must
not rerun DEV1 merely to recreate timing.
