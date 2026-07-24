# Stage 9.3.5 Projection Feasibility and Causal Closure

Stage 9.3.5 is a bounded, audit-only diagnostic for the current same-lineage
Stage 9.3.4 baseline. It consumes the GRAB `s1/airplane_lift` / right /
`artimano_rh` 60-frame baseline and the existing causal frame selection. It
does not rewrite Eq. (1)--(9), paper weights, Stage 7 warm-start, Stage 8
graph, Stage 9.2 historical artifacts, Stage 10 manifests, or manual
acceptance.

## Contracts

The projection state metric is versioned as
`toporetarget.projection_state_metric.v1`. It uses the formal regularization
scales with the current-frame warm state as its centre. This is explicitly
`diagnostic_only`, `paper_method=false`, and `accepted_reference=false`; it is
not the paper objective. The official final is treated as a known feasible seed
only after an independent full-512 `reference_triangle_winding` validation.

The warm-to-final path uses linear qpos/translation interpolation and an
SO(3) Exp/Log geodesic. It samples at least 1001 alphas, records all feasible
intervals, refines boundaries to 1e-6, and does not assume monotonicity. The
minimal soft-safe and official-slack projections use all 512 collision samples
from their first solver call. A pressure score is a transparent engineering
diagnostic, not an SLSQP dual multiplier. Counterfactual states are allowed to
be infeasible and are never written as formal artifacts.

## Commands

```bash
export PYTHONNOUSERSITE=1 PYTHONPATH=src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/deepcybo/miniconda3/envs/topo-retarget/bin/python
MANIFEST=.local/runs/stage9_3_4_current_lane/baseline/current_lineage_manifest.json
BASELINE=.local/runs/stage9_3_4_current_lane/baseline/current_lineage_baseline.zarr
RUN=s1__airplane_lift__right__artimano_rh__f000240_f000300

$PY -m toporetarget workflow scan-warm-final-feasibility \
  --current-lineage-manifest "$MANIFEST" --current-baseline "$BASELINE" \
  --frames auto --samples 1001 --resume \
  --output-root ".local/runs/stage9_3_5_projection/$RUN"

$PY -m toporetarget workflow run-feasibility-projection \
  --current-lineage-manifest "$MANIFEST" --current-baseline "$BASELINE" \
  --path-scan-root ".local/runs/stage9_3_5_projection/$RUN" \
  --output-root ".local/runs/stage9_3_5_projection/$RUN" \
  --profiles minimal_soft_safe_projection_from_warm_v2,official_slack_projection_from_warm_v2 \
  --full-512 --resume

$PY -m toporetarget workflow run-state-counterfactuals \
  --current-lineage-manifest "$MANIFEST" --current-baseline "$BASELINE" \
  --projection-root ".local/runs/stage9_3_5_projection/$RUN" \
  --output-root ".local/runs/stage9_3_5_counterfactual/$RUN"

$PY -m toporetarget workflow attribute-objective-constraints \
  --current-lineage-manifest "$MANIFEST" --current-baseline "$BASELINE" \
  --projection-root ".local/runs/stage9_3_5_projection/$RUN" \
  --counterfactual-root ".local/runs/stage9_3_5_counterfactual/$RUN" \
  --output-root ".local/runs/stage9_3_5_objective_attribution/$RUN" \
  --constraint-output-root ".local/runs/stage9_3_5_constraint_attribution/$RUN"

$PY -m toporetarget workflow run-projection-branch \
  --current-lineage-manifest "$MANIFEST" --current-baseline "$BASELINE" \
  --projection-root ".local/runs/stage9_3_5_projection/$RUN" \
  --output-root ".local/runs/stage9_3_5_branch_rollout/$RUN"
```

The final report and HTML are assembled with
`stage9-causal-closure-status`. All results belong under `.local/runs/stage9_3_5_*`
or `.local/reports/stage9_3_5/`. A branch rollout is `NOT_REQUIRED_BY_GATE`
when no projection candidate passes the explicit multi-frame gate; that status
is not a failure.

The generated HTML is self-contained and has frame/state selectors, warm/projection/final
switching, an alpha slider, feasible-interval and per-finger RMSE plots, objective endpoint/
path/variable-group tables, base-versus-q counterfactuals, per-link/per-finger pressure filters,
interaction-gradient columns, projection attempts, branch status, and root-cause/readiness panels.
Its displayed scales are fixed from the complete report payload. `official_artifact_immutability.json`
records SHA-256 and mtime comparisons for the Stage 5--10/current-lineage boundary; any change
fails the readiness gate closed.

## Interpretation boundary

Directional and path-integrated attribution is numerical local/path evidence,
not a complete game-theoretic causal proof. The final route must retain
`ENTER_STAGE9_4=NO`, `HUMAN_DECISION_REQUIRED=YES`, and
`STOP_AFTER_STAGE9_3_5=TRUE` until a human approves one Stage 9.4 direction.

The one-shot closure that followed this decision boundary is recorded in
[`STAGE9_ONE_SHOT_CAUSAL_CLOSURE_AND_REPAIR.md`](STAGE9_ONE_SHOT_CAUSAL_CLOSURE_AND_REPAIR.md).
It preserves this diagnostic projection boundary and does not treat projection
as a paper-method or accepted-reference result.
