# Stage 9.3.1 signed-distance reconciliation and bounded shadow ablation

Stage 9.3.1 is a read-only reconciliation boundary over the accepted Stage
9.2/Stage 10 reference runtime and the existing Stage 9.3 contact audit. It
does not change Eq. (1)-(9), paper weights, the Stage 9.2 solver profile, the
accepted final artifact, the Stage 10 manifest/export, or manual acceptance.
It never runs a 60-frame optimizer.

## Commands

```bash
conda run -n topo-retarget env PYTHONNOUSERSITE=1 PYTHONPATH=src \
  python -m toporetarget workflow reconcile-contact-metrics \
  --run .local/runs/stage10_reference_runtime/<run>/manifest.json \
  --contact-audit-root .local/runs/stage9_3_contact_audit/<run> \
  --output-root .local/runs/stage9_3_1_metric_reconciliation/<run> --force

conda run -n topo-retarget env PYTHONNOUSERSITE=1 PYTHONPATH=src \
  python -m toporetarget workflow run-contact-shadow-ablation \
  --run .local/runs/stage10_reference_runtime/<run>/manifest.json \
  --reconciliation-root .local/runs/stage9_3_1_metric_reconciliation/<run> \
  --output-root .local/runs/stage9_3_shadow_ablation/<run> \
  --frames auto \
  --profiles official_baseline_reproduction,half_active_margin,zero_active_margin,full_512_query_reference,minimal_soft_safe_projection_from_warm,official_slack_projection_from_warm
```

The reconciliation command resolves and records every available input path,
SHA-256, mtime, profile, sequence, hand, robot, and frame range. The shadow
command is fail-closed: it may write only its isolated diagnostic bundle and
must report `solver_invocation_count: 0` unless the reconciliation gate passes.
The current implementation does not implicitly turn a diagnostic request into
a solver run.

## Metric contract

All signed distances use positive-outside convention and metres. The formal
Stage 9.2 `full_signed_distance` is the persisted 512-point value from the
reference triangle/winding backend. `max_penetration` is the raw diagnostic
`max(max(-min(phi_full)), 0)`; it is not tau-adjusted, clipped for display, or
replaced by slack. The soft and hard constraints remain, respectively,
`phi + tau >= 0` and `phi + b >= 0` with the persisted queried slack contract.

Stage 9.3.1 separately compares that reference value with the legacy
`convex_hull_exact_solver_only` value used by the old Stage 9.3 report. A
negative legacy value is not evidence against the Stage 9.2 reference
acceptance until both values use the same SDF definition.

## Required outputs

The reconciliation directory contains `input_identity_audit.json`,
`signed_distance_definition_matrix.{json,md}`,
`full512_identity_comparison.json`, `full512_distance_reconciliation.{json,csv}`
and mismatch records, `transform_chain_comparison.csv`,
`acceptance_replay.{json,csv}`, `collision_offset_direction_audit.json`,
`collision_offset_per_link.csv`, `metric_reconciliation_summary.json`,
`metric_reconciliation_summary.md`, `shadow_frame_selection.json`,
`metric_reconciliation_and_shadow.html`, and `audit_manifest.json`.

The input identity audit reports the declared schema for each artifact. Legacy
NPZ/report files that do not contain a schema marker are recorded explicitly as
`unversioned:<filename>`; the audit does not invent a paper schema for them.

The shadow directory contains `shadow_manifest.json`,
`shadow_frame_selection.json`, `shadow_profiles.json`, per-frame/per-profile
result placeholders, causal-analysis outputs, comparison outputs, and
`stage9_4_readiness.json`. A failed gate means all requested profiles are
`not_run`, `diagnostic_only: true`, `paper_method: false`,
`accepted_reference: false`, and Stage 9.4 is not entered.

## Current accepted-window result

For `s1/airplane_lift`, right Arti-MANO, global `[240,300)` / local `[0,60)`:

- 512-point identity, ordering, transform chain, and Stage 9.2 reference
  replay pass; persisted-versus-reference maximum difference is below
  `2.5e-16 m`.
- Independent acceptance replay is `60/60`, with zero formal/replay
  mismatches.
- The legacy Stage 9.3 backend does not reconcile with the reference backend;
  the largest absolute difference is `19.485 mm` and the sign mismatch count
  is `180`.
- The directional visual/collision offset audit is
  `COLLISION_VISUAL_OFFSET_DIRECTION_INCONCLUSIVE` because the available visual
  meshes are open and unsigned distance cannot prove outward inflation.
- The unique closeout state is
  `RETURN_TO_STAGE9_2_ACCEPTANCE_OR_METRIC_FIX`. No shadow profile ran and no
  solver invocation occurred.

These results are diagnostic evidence only. They do not alter the accepted
Stage 9.2 Zarr, Stage 10 manifest, exports, or manual acceptance record.
