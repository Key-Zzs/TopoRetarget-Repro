# ContactPose mug solver-feasibility closeout

Status: `RESOLVED` for the frozen static `contactpose:full1_use/mug` Stage-12
selection. This is a solver-engineering closeout, not a reproduction of the
paper's ContactPose contact benchmark.

The frozen rejected result had SLSQP status 8 (`Positive directional derivative
for linesearch`). Its geometric full-surface audit passed, but the active
constraint state did not: six soft residuals were below `-1e-6` because the
state carried stale/insufficient slack after an active-set continuation. The
same primary q/base state had positive hard residuals and a representable slack
repair; therefore this is classified as `BOOKKEEPING_OR_STATE_MISMATCH`, not
true local infeasibility or a ContactPose/mug-specific condition.

The repair is generic. On a recoverable status 8/9 exit, it re-evaluates the
original active constraints at the returned fixed q/base state, rejects any
negative hard residual, and raises only each required slack coordinate to the
next representable float satisfying the unchanged original inequality. It then
runs the existing bounded reference-batched SLSQP path and the unchanged strict
acceptance/full-audit gates. It does not change objective terms, constraint
rows, query semantics, bounds, tolerances, penalties, collision samples, or
dataset/object parameters.

The immutable rejected diagnostic remains retained separately. Evidence lives
under `.local/reports/contactpose_mug_solver_repair/` and the frozen input copy
under `.local/archive/contactpose_mug_status8_baseline_20260731T000000Z/`.
