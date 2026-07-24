# Stage 9 One-Shot Causal Closure and Repair

This document records the bounded Stage 9.3.6--9.4 closure for the frozen
`GRAB/s1/airplane_lift`, right-hand `artimano_rh`, global-frame `[240, 300)`
lineage. The current-lineage hash, source, warm-start, graph, collision, and
historical/current-final identities are recorded in
`.local/reports/stage9_one_shot/input_identity_and_immutability.json`.

## Contract

Projection is diagnostic-only: `diagnostic_only=true`, `paper_method=false`,
and `accepted_reference=false`. Its final contract permits only
`ANALYTIC_IDENTITY_PROJECTION`, `SOLVED_AND_VALIDATED`,
`FEASIBLE_UPPER_BOUND_ONLY`, and `INVALID_CONTRACT`; it is not a future gate
and no further projection ablation is required.

The formal Eq. (9) audit is separate from projection. The implementation map
is in `formal_regularization_code_map.json` and `.md`. It records that the
current temporal term regularizes the six-dimensional base correction and the
finger correction together, while the paper specifies the temporal `q_theta`
term separately from the base position and rotation priors. Previous final
states are remapped into the current seed chart before comparison.

## Fixed causal sweep

Exactly the declared selected frames `(0, 10, 30, 36, 39)` and profiles C0--C7
are recorded in `decisive_ablation_results.csv` and
`decisive_ablation_summary.json`. Profiles that cannot produce a strict
candidate remain recorded as failures; they are not silently converted into
passes. `contact_proxy` and `contact_retention_proxy` remain diagnostic
proxies, not contact ground truth.

The decisive root-cause file is `root_cause_final.json`. It chooses one primary
cause, `IMPLEMENTATION_REGULARIZATION_BUG`, with base motion and temporal
objective pressure retained only as secondary factors.

## Single repair

The only repair candidate is `faithful_regularization_fix_v1`, using solver
profile `scipy_slsqp_active_set_contact_rich_v3_fixed`. It removes base motion
from the temporal `q` term while retaining the paper-specified base priors and
all formal weights. The full result is isolated under
`.local/runs/stage9_4/faithful_regularization_fix_v1/`; the old final and old
Stage 10 artifacts are preserved.

Full-run validation, comparison metrics, bounded regression, final decision,
and the versioned Stage 10 review bundle are generated under
`.local/reports/stage9_one_shot/` and
`.local/runs/stage10_faithful_regularization_fix_v1/`. Human manual acceptance
remains a separate gate from the machine validation; that later gate passed as
case A in Faithful Reproduction Finalization.

## Reproduction and verification

Use the repository's validated `topo-retarget` interpreter:

```bash
/home/deepcybo/miniconda3/envs/topo-retarget/bin/python -m toporetarget.cli.workflow stage9-one-shot
```

The run is intentionally isolated and does not add, commit, push, reset, or
clean Git state. The machine-readable final status is
`.local/reports/stage9_one_shot/stage9_final_decision.json`.

For the executed lineage, the full 60-frame and collision gates passed, but the
repair quality gate rejected the candidate because mean long-finger RMSE
increased by `0.0305 mm` while the safeguarded threshold required `1.5392 mm`
improvement. The final machine status is therefore
`REPAIR_CANDIDATE_REJECTED`; the current-lineage baseline remains recommended.

## Faithful reproduction finalization note

The rejection above belongs to the Stage 9 improvement gate: the candidate did
not achieve the required `1.5392 mm` long-finger improvement. A subsequent
four-state, 60-frame visual/numerical review found no visible old-to-fixed
degradation and classified the v3 fixed profile as quality-neutral. The
canonical paper-faithful/legacy profile split and the isolated fixed Stage 10
candidate are documented in
[`FAITHFUL_REPRODUCTION_FINALIZATION.md`](FAITHFUL_REPRODUCTION_FINALIZATION.md).
The repository-valid human signature has passed with decision case A.
