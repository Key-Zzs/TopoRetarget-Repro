# Collision QuerySet and slack contract

Stage 9 uses the immutable Stage 6 robot collision surface artifact. For the supplied
Arti-MANO assets this is 32 deterministic samples per collision geometry, 512 samples
total. Visual samples, object samples, contact labels, and generated tip points are not
used as optimization collision queries.

## QuerySet profiles

`full_collision_surface_reference_v1` includes all 512 samples. It is the reference
profile and is used for independent validation.

`adaptive_active_set_v1` starts with every sample inside the 10 mm active margin,
initial penetrations, and the nearest sample from every collision geometry. After each
solve, every newly violated or active full-surface sample is added monotonically. The
outer loop stops when no new sample is added or after five rounds. Duplicate IDs are
rejected, IDs are sorted deterministically, and the inclusion reason and round are
stored in the artifact.

## Signed-distance constraints

The reference SDF is positive outside the closed object mesh. With `tau=1 mm` and
`b=30 mm`, the solver uses per-query slack to make the soft tolerance explicit while
retaining the hard bound:

```text
phi_i(q) >= -b
phi_i(q) + s_i >= -tau
s_i >= 0
s_i <= b - tau
```

The active-set loop uses the validated solver backend after probe comparison, while the
final report recomputes all 512 reference distances independently. It records
minimum distance, hard/soft violations, maximum and total slack, and the number of
full-surface samples that were not in the active QuerySet. A bounded result must not
hide a missed full-surface violation behind the adaptive set.

Slack is initialized per selected query as

```text
s_i^0 = clip(max(-tau - phi_i(x_0), 0), 0, b - tau)
```

The hard residual never includes slack; the soft residual is `phi_i + s_i + tau`.
The Stage 6 asset reports fixed fingertip visual spheres without collision geometry.
Stage 9 does not silently convert them into collision samples, so the artifact records
`visual_collision_fallback=false` and this coverage limitation explicitly.

Useful commands are `inspect-query-set`, `compare-query-profiles`, `refine`,
`validate-refinement`, `audit-penetration`, and `compare-solvers`. If a strict run
fails, inspect the frame/status message, query hash, reference sign validity, hard and
soft residuals, and active-set round before changing any profile. Do not solve a failure
by raising `b`/`tau`, removing samples, switching to unsigned distance, or using visual
geometry.

## Provenance and audit

The final Zarr metadata records the robot-surface hash, object mesh hash, source cache
hash, graph hash, warm-start hash, profile hashes, paper weights, solver-only SDF
cross-validation, and the explicit no-Stage-10/RL/physics provenance boundary. The
source and Stage 6 artifacts remain read-only; derived reports belong under ignored
`.local/reports/stage9/`.

## Stage 9.1 continuation and termination contract

When the adaptive QuerySet grows, v2 continues from the previous optimizer
result rather than reconstructing the Stage 7 warm seed. Base/q coordinates are
copied from `result.x`, existing slack values are looked up by stable query ID,
and only newly added IDs receive the minimum bounded slack initialization. The
set grows monotonically and the before/after IDs plus the continuation policy
are persisted in provenance. v1 retains its historical warm-seed
reinitialization policy for regression comparison.

Feasibility does not imply solver convergence: status `9` is recorded as a
non-converged optimizer result and cannot pass strict acceptance even when all
q/slack, active-query, and full 512-point audits are feasible. The v2 profile
uses the strict acceptance policy and defers the separate stationarity policy;
the fixed-grid benchmark and repeat evidence are in
`.local/reports/stage9_1/maxiter_benchmark.json`.
For the current Stage 10 inputs, the source/coordinate audit classifies
`[240,300)` as `contact_rich`, `[238,298)` as `approach`, and the RH regression
as `pre_contact`. The bounded far-vs-contact comparison is in
`.local/reports/stage9_solver_closeout/far_vs_contact_solver_comparison.json`;
semantic contacts remain outside the Eq. (8)/(9) objective.

## Stage 9.3.3 numerical gate

The shadow-equivalence workflow compares the official final QuerySet IDs/order,
strict status, bounds, hard/soft/full-512 audits, state arrays, and objective
components before any causal profile is allowed to run. Its canonical SDF is
the same reference-winding definition used by Stage 9.3.2. A feasible replay
with a different state is not accepted as numerical equivalence, and status 9
is never normalized to success. All shadow checkpoints are diagnostic-only and
isolated under `.local/runs/stage9_3_3_shadow_*`.

## Stage 9.3.5 diagnostic projection boundary

The Stage 9.3.5 feasibility scan and projection use the same canonical
reference-winding SDF and all 512 collision samples. They may report
`projection_state_metric` values and bounded slack diagnostics, but these
states are not QuerySet solver results and are never eligible for formal
acceptance. The official full-512 QuerySet, slack bounds, and strict
acceptance contract remain unchanged.
