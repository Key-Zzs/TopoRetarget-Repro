# Stage 9.3 contact-retention and collision-geometry audit

Stage 9.3 is an audit-only workflow over an accepted Stage 9.2/Stage 10
manifest. It reads the manifest-resolved canonical, warm-start, final,
interaction-graph, object-sample, and collision-sample artifacts. The default
path never calls the Stage 9 optimizer and never writes a formal input artifact.

Run the complete accepted window with:

```bash
conda run -n topo-retarget env PYTHONNOUSERSITE=1 \
  python -m toporetarget workflow audit-contact-retention \
  --run .local/runs/stage10_reference_runtime/<run>/manifest.json \
  --output-dir .local/runs/stage9_3_contact_audit/<run> \
  --surface-samples 8192 --thresholds-mm 1,2,3,5,8,10 --html --force
```

The audit enforces the complete 60-frame acceptance range. It records the
resolved inputs, hashes and mtimes before/after, branch and HEAD, solver and
execution profiles, signed-distance convention, surface-sampling profile, and
all generated output hashes in `audit_manifest.json`.

## What is compared

The report keeps four geometric roles separate:

- **Source:** the canonical deforming source hand mesh and its 21 MediaPipe
  anchors.
- **Warm-start:** the Arti-MANO Stage 7 initialization loaded from the formal
  warm-start artifact.
- **Final:** the accepted Stage 9.2 final trajectory.
- **Object:** the canonical watertight object mesh transformed from object-local
  coordinates into the scene frame for every frame.

Source contact is a diagnostic proxy, not ground truth contact. The proxy uses
deterministic dense mesh samples, nearest MediaPipe21 anchor-region labels, and
the signed-distance threshold sweep. Robot contact retention is reported at
semantic anchors, per hand region, and source/final threshold pairs. The
semantic anchors are not a pad surface; the audit therefore also evaluates
visual robot mesh samples and the independent 512-point collision geometry.

Signed distance is positive outside and negative inside. Dense surface values
are an approximation to the continuous surface and always carry that label.
The final artifact's requested convex-hull solver backend is selected only
after 32 deterministic probe queries cross-validate it against the reference
triangle/SDF backend. The audit records both backends and the cross-validation
result; it does not silently substitute a mesh or repair geometry.

`queryset_audit_per_point.csv` preserves query point IDs, source/robot links,
object-local coordinates, active margin, inclusion reason, expansion state,
slack, and warm/final distances. `queryset_audit_per_link.csv` and
`queryset_audit_per_frame.json` provide the link/frame summaries. SciPy SLSQP
multipliers are recorded as unavailable; slack, active-set provenance, and
independent full-surface checks are used instead.

## Objective and path diagnostics

`objective_tradeoff_per_frame.csv` evaluates the same Stage 9 objective
definition for warm-start and final states, including raw and weighted
interaction/bone terms, base regularization, temporal regularization, slack,
and the total. It does not introduce a contact-preservation term or alter the
formal objective.

`warm_final_interpolation_per_frame.csv` is a counterfactual diagnostic path:
qpos is linearly interpolated and base rotation uses SO(3) Slerp. It is not an
optimizer trajectory, does not imply feasible intermediate states, and does
not change the final artifact. `contact_retention_proxy.json` includes
anchor-level distance drift, object-local direction consistency, and threshold
sensitivity; `per_link_collision_visual_offset.csv` reports the deterministic
bidirectional visual/collision surface-sample offset approximation.

`trajectory_contact_audit.html` is self-contained and supports source,
warm-start, final, object, visual, collision, QuerySet, anchor, segment,
threshold, frame, and link/region controls. It is an inspection aid; the CSV
and JSON numeric reports remain authoritative.

## Shadow ablation and interpretation

The default audit is the formal no-solver path. `--run-shadow-ablation` is a
separate diagnostic boundary and is never accepted as paper-faithful evidence.
When requested, the current implementation performs score-only counterfactual
decomposition on selected frames (removing persisted slack, temporal, or base
score terms); it does not regenerate q or identify a causal optimizer effect.
If no shadow run is present, `shadow_ablation_status.json` explicitly records
that the counterfactual evidence is missing. A shadow result cannot replace or
overwrite the accepted Stage 9.2/Stage 10 artifacts.
Explicit shadow output is isolated under `.local/runs/stage9_3_shadow_ablation/`.

The root-cause report separates geometry inflation, collision-sample coverage,
semantic-anchor/pad mismatch, QuerySet activation, and objective/regularization
explanations. Each item includes confidence, evidence for and against, and a
next diagnostic action. Audit statuses intentionally distinguish numerical
validity, collision feasibility, contact retention, visual clearance, source
contact richness, temporal continuity, and physical trackability.

## Assumptions and provenance

The following are engineering diagnostics rather than claims about the paper:
`A_STAGE9_3_DENSE_SURFACE_APPROXIMATION_001`,
`A_STAGE9_3_CONTACT_PROXY_001`, `A_STAGE9_3_PAD_PROXY_001`,
`A_STAGE9_3_INTERPOLATION_001`, and
`A_STAGE9_3_SLSQP_MULTIPLIER_001`. They are listed in `docs/ASSUMPTIONS.md`
and embedded in the audit manifest. The formal Stage 9.2 profile, weights,
query-set definition, 512 samples, and strict acceptance policy remain
unchanged.
