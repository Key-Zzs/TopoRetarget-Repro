# Stage 9.3.2 canonical contact distance and re-audit

Stage 9.3.2 is an audit-only boundary over the accepted Stage 9.2 and Stage
10 reference runtime. Its formal evaluation backend is the strict
`reference_triangle_winding` SDF, with positive-outside sign, object-local
queries transformed from the scene frame, exact triangle-mesh closest points,
and strict sign validity. The versioned contract is
[`configs/audit/contact_distance/reference_winding_v1.yaml`](../configs/audit/contact_distance/reference_winding_v1.yaml).

The solver backend and the formal evaluation backend are separate contracts.
Stage 9.2 may retain the solver profile's approved convex-hull acceleration,
but every formal penetration, contact proxy, visual distance, collision
distance, full-512 result, HTML value, root-cause conclusion, and shadow
evaluation is recomputed with the reference winding backend. The legacy
`convex_hull_exact_solver_only` report remains available for regression and
historical explanation only.

## Audit contract

The v2 audit records the canonical profile and hash, solver and legacy backend
identities, sign convention, units, object mesh hash, coordinate frame,
source/warm/final artifact identities, deterministic dense-surface sampling,
collision sampling, thresholds, and legacy comparison status. Raw signed
distance, raw penetration, penetration beyond `tau`, hard-bound violation,
soft residual before/after slack, and hard residual are separate fields. The
compatibility field `max_penetration` is retained with an explicit
`legacy_metric_semantics` description.

Dense visual values are deterministic surface-sample approximations; they are
not exact triangle-to-triangle distances. Source and robot contact are named
`source_contact_proxy` and `contact_retention_proxy`, never ground truth. A
semantic anchor is a skeleton anchor and is not guaranteed to lie on a visual
contact surface.

The collision/visual audit reports unsigned bidirectional coverage gaps and
per-link statistics. Because the available visual meshes are open or have
unvalidated normals, it reports `offset_direction: INCONCLUSIVE` and cannot
support `COLLISION_GEOMETRY_INFLATED` or `inset` claims from unsigned offsets.

## Reconciliation and gate

The canonical re-audit first reconciles all 60 x 512 final collision samples
against persisted Stage 9.2 values and the independent validator. The formal
gate requires identity and transform equality, maximum distance difference at
most `1e-10 m`, zero sign mismatches, an accepted replay, complete canonical
source/warm/final reports, explicit source classification, quantified legacy
disagreement, and unchanged official artifacts. A failed gate returns
`RETURN_TO_STAGE9_2_ACCEPTANCE_OR_VALIDATION_FIX` and prevents shadow solving.

Only after the gate passes may the bounded shadow run select at most three
deterministic frames. Its six profiles are diagnostic-only, paper-external,
and isolated from formal artifacts: official baseline, half margin, zero
margin, full-512 QuerySet, minimal soft-safe projection, and official-slack
projection. Shadow output can support engineering hypotheses about geometry,
coverage, QuerySet, or margin, but it is never an accepted reference and does
not start Stage 9.4 automatically.

## Stage 9.4 and Stage 10 boundaries

`stage9_4_readiness.json` must use one of the versioned readiness states and
must identify evidence for and against each root-cause candidate. Geometry,
coverage, anchor, QuerySet, and margin evidence must be separated from any
paper-objective hypothesis. Stage 9.4 implementation is outside this run;
the next run must name the exact module and preserve Eq. (1)-(9), paper
weights, and the accepted solver contract.

This audit does not modify the Stage 9.2 final, repeat, checkpoints, solver or
execution profiles, Stage 10 manifest, manual acceptance, reference-runtime
acceptance, or robot reference export. Audit solver invocation count is zero;
any shadow solver invocations are recorded only under the shadow root.

## Commands

```bash
export PYTHONNOUSERSITE=1 PYTHONPATH=src
/home/deepcybo/miniconda3/envs/topo-retarget/bin/python \
  -m toporetarget workflow reaudit-contact-canonical \
  --run .local/runs/stage10_reference_runtime/<run>/manifest.json \
  --legacy-audit-root .local/runs/stage9_3_contact_audit/<run> \
  --reconciliation-root .local/runs/stage9_3_1_metric_reconciliation/<run> \
  --output-root .local/runs/stage9_3_2_canonical_reaudit/<run> \
  --surface-samples 8192 --html --force
```

The independent legacy report is never overwritten. The v2 JSON, CSV, and
self-contained `trajectory_contact_audit_v2.html` are the formal audit
outputs; HTML defaults to canonical values and labels legacy values
`DIAGNOSTIC ONLY / SUPERSEDED`.

## Stage 9.3.3 shadow gate

Stage 9.3.3 consumes this canonical audit only after the formal baseline
reproduction passes. It records source/warm/official/shadow per-finger metrics,
full-512 canonical SDF, QuerySet and constraint attribution, while retaining
`source_contact_proxy` and `contact_retention_proxy` as proxies. A failed
baseline returns `RETURN_TO_STAGE9_3_2_SHADOW_HARNESS_FIX` and runs zero
mandatory shadow profiles.
