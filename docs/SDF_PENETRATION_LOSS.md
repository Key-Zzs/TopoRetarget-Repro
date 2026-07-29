# Dense SDF penetration loss

This branch adds the paper-external `dense_sdf_penetration` objective term.
It is deliberately additive: the paper objective, paper weights, active-set
constraints, slack variables, and full-surface audits are unchanged. The SDF
term is not the Eq. (8) collision constraint and its loss dead zone must not be
interpreted as a change to the Eq. (8) hard/soft tolerances, slack, `tau`, or
`b`.

## Versioned profiles

The initial S1 profile was
`configs/retarget/penetration_losses/dense_squared_hinge_v1.yaml`:

\[
 E_{SDF}=\frac{1}{|G|}\sum_g\frac{1}{|g|}
 \sum_{i\in g}\left(\frac{\max(-\phi_i,0)}{d_{ref}}\right)^2,
 \qquad d_{ref}=1\,\mathrm{mm}.
\]

It used a zero loss dead zone (`penetration_tolerance_m=0`) and is retained
for zero-tolerance comparison only, with
`deprecated_for_zero_tolerance_comparison=true`. This zero-tolerance choice
was not equivalent to the tolerance semantics already present in Eq. (8).

The active S1 profile is
`configs/retarget/penetration_losses/dense_squared_hinge_deadzone1mm_v2.yaml`.
For positive-outside signed distance, it defines

\[
 d_i=\max(0,-\phi_i),\qquad
 e_i=\max(0,d_i-1\,\mathrm{mm}),\qquad
 z_i=e_i/1\,\mathrm{mm},
\]
\[
 E_{SDF}=\frac{1}{|G|}\sum_g\frac{1}{|g|}
 \sum_{i\in g}z_i^2.
\]

Thus penetration up to and including 1 mm contributes zero; only the excess
penetration contributes. This is a loss dead zone, not a collision-constraint
tolerance. The Eq. (8) hard/soft constraints, slack, `tau`, `b`, and the
existing penetration acceptance gate remain unchanged.

`phi` uses the existing positive-outside signed-distance convention. Samples
are the complete 512-point Arti-MANO collision surface (16 geometries × 32
samples); there is no clearance offset, slack term, contact attraction, or
normalization by the number of frames.

## Backend split

The optimizer inner loop uses the strict solver-only convex-hull backend after
its existing cross-validation. It supplies the objective value, SDF gradient,
and SDF-loss callbacks and is recorded as `sdf_loss_backend=solver_fast_backend`
(`convex_hull_exact_solver_only` in artifact-level backend IDs). The independent
Stage 6 reference winding backend remains mandatory for the final full-surface
audit, penetration validation, and acceptance report and is recorded as
`validation_sdf_backend=reference_winding_v1`.
For the frozen S1 G1/G2 meshes, the artifact-level validation backend is
`reference_triangle_winding`; the open-object hybrid backend is not substituted
into this validation path.

The final validation path must never be replaced by the fast backend. Gradients
use signed-distance surface normals multiplied by the analytic collision-point
Jacobian. A finite-difference fallback is recorded if a query is non-smooth or
invalid; it never changes the paper constraint system. No triangle-level
reference-winding query is allowed inside an objective callback.

With `--lambda-sdf 0`, the term contributes exactly zero to the optimizer
objective and gradient. A cheap independent full-surface diagnostic still
persists `e_sdf` and `weighted_e_sdf=0`, so the E0-vs-S1 comparison has a real
baseline without changing the E0 trajectory.

The v2 migration is audited under
`.local/experiments/s1_sdf_penetration_loss_v1/reports/` by
`sdf_loss_profile_migration.json`, `backend_split_validation.json`, and
`smoke_v2.json`. The full v2 experiment itself is isolated under
`.local/experiments/s1_sdf_penetration_loss_v1/v2_deadzone1mm/` so the original
v1 artifacts and failed/incomplete solver records remain intact.

Example:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src python -m toporetarget retarget refine \
  --canonical .../canonical.zarr --warm-start .../warm_start.npz \
  --graph .../interaction_graph.npz \
  --collision-samples .../artimano_rh_collision_surface.npz \
  --lambda-sdf 0.1 --penetration-loss-profile dense_squared_hinge_deadzone1mm_v2 \
  --solver-profile scipy_slsqp_active_set_contact_rich_v3_fixed \
  --execution-profile cached_checkpoint_cpu_float64_v3
```

S1 remains a comparison only. It cannot change the paper baseline or claim
paper-level contact-retention improvement.

## S1.1 signal-rich follow-up

The broader signal-rich evaluation is a separate experiment under
`.local/experiments/s1_1_signal_rich_grab_v1/`. Its source-only stratification,
backend gate, and fail-closed decision contract are specified in
`docs/S1_1_SIGNAL_RICH_GRAB_EVALUATION.md`; the old two-clip S1 artifacts and
profile remain preserved.

## S1.2A E0 penetration stress discovery

S1.2A is the bounded E0-active stress-set lane documented in
`docs/S1_2A_E0_PENETRATION_STRESS_DISCOVERY.md`. It selects candidates from
source-only GRAB eligibility, uses a fixed three-frame warm/E0 funnel, freezes
the top three by E0 robot penetration, and then evaluates E0 versus the
unchanged S1 profile on the same full 60-frame inputs. It does not use S1
results for selection, alter the Eq. (8)/(9) constraints, or change the global
default. A passing result is only
`S1_CONDITIONALLY_ACCEPTED_ON_STRESS_SET` and remains stress-set scoped.
