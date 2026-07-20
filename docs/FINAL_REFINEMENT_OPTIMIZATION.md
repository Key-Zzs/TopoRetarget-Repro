# Stage 9 final refinement optimization

Status: `implemented_with_assumptions` for the bounded RH/LH `s7/cubemedium_inspect_1`
window, frames `[0, 60)`.

## Objective

For each frame, Stage 9 refines the Stage 7 robot pose while keeping the frozen Stage 8
source interaction graph and weights. The implementation evaluates the paper's Eq. (8)
and Eq. (9) with the paper values loaded from `configs/paper/retarget.yaml`:

```text
E_8 = lambda_IM E_IM + lambda_bone E_bone + lambda_reg E_temporal
      + lambda_base_pos E_base_pos + lambda_base_rot E_base_rot
      + 0.5 w_s sum_i s_i^2

min_x E_8
subject to  phi_i(q) >= -b,
            phi_i(q) + s_i >= -tau,
            s_i >= 0,
            q_min <= q <= q_max
```

`E_IM` is the exact mean-square Eq. (7) residual, `E_bone` is the raw Eq. (1)
sum, `E_temporal` compares the current pose to the previous final pose, and the
base priors are corrections relative to the current seed. The positive-outside SDF
convention is retained throughout.

## Coordinates and temporal policy

The default profile is `local_seed_delta_v1`:

```text
R_base = Exp(delta_omega) R_seed
p_base = p_seed + delta_p
x = [delta_p(3), delta_omega(3), q_theta(22), s(|Q_t|)]
```

The first frame starts from the warm-start seed with zero delta and no temporal
residual. Later frames initialize from the previous final pose remapped into the
current seed coordinates. Translation is metres, rotation and joint coordinates are
radians, and no hidden normalization or FPS scaling is applied.

## Solver and derivatives

The engineering solver is float64 SciPy SLSQP with analytic Torch-autograd objective
derivatives and SDF-normal times collision-point Jacobians. Invalid SDF-normal rows
fall back to central finite differences and are counted. URDF q bounds, non-negative
slack bounds, fail-fast solver status, and the configured `maxiter=30`, `ftol=1e-7`
are recorded in the final artifact. These solver details are not disclosed by the
paper and are not presented as paper facts.

The solver may use `convex_hull_exact_solver_only` for the closed convex cubemedium
mesh after 32 deterministic probe comparisons against the Stage 6 reference backend.
Initial QuerySet selection and active-set expansion use that validated solver backend;
the persisted per-frame full-surface distances and final acceptance audit always query
the Stage 6 reference backend.

## Failure policy

An unsuccessful SLSQP call, non-finite objective/constraint, violated q bound, invalid
artifact shape, source/hash mismatch, or failed independent audit stops the command.
No solver failure is converted into a successful frame, and no Stage 10, RL, physics,
ContactPose, or baseline behavior is included.

See [`COLLISION_QUERY_SET_AND_SLACK.md`](COLLISION_QUERY_SET_AND_SLACK.md) for QuerySet
construction and [`stages/STAGE_9_FINAL_CONSTRAINED_REFINEMENT.md`](stages/STAGE_9_FINAL_CONSTRAINED_REFINEMENT.md)
for the bounded acceptance procedure.
