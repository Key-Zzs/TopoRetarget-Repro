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
radians, and no FPS or dt scaling is applied. The execution layer may use the
explicit, invertible `seed_delta_normalized_v1` map internally for SLSQP
conditioning; callbacks, audits, and persisted artifacts always use the raw vector above.

## Solver and derivatives

The engineering solver is float64 SciPy SLSQP with analytic Torch-autograd objective
derivatives and SDF-normal times collision-point Jacobians. Invalid SDF-normal rows
fall back to central finite differences and are counted. URDF q bounds, non-negative
slack bounds, fail-fast solver status, and the configured `maxiter=100`, `ftol=1e-7`
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

## Stage 9.1 solver-robustness closeout

The contact-rich closeout preserves
`scipy_slsqp_active_set_v1.yaml` unchanged and registers the independent
`scipy_slsqp_active_set_contact_rich_v2.yaml` profile. The v2 active-set
continuation starts the next SLSQP call from the preceding `result.x`: base
correction and qpos are copied directly, old slack is remapped by query ID, and
new slack is initialized as
`clip(max(-tau - phi_i(result.x), 0), 0, b - tau)`. Query IDs are never removed
or silently reinterpreted, and each continuation decision is recorded in the
artifact provenance.

Solver termination is separate from primal feasibility. A feasible candidate
with SciPy status `9` / `Iteration limit reached` remains `accepted=false` under
the strict policy. Strict acceptance requires optimizer convergence, q/slack
bounds, active constraints, the independent 512-point hard and soft audits,
active-set convergence, and finite values. The artifact stores the optimizer
status/message/iteration and evaluation counters, objective change, step norm,
all individual checks, acceptance policy ID, and reason. The optional
`feasible_stationary_v1` policy is deferred; no status-9 relaxation is enabled.

The fixed benchmark grid is `[30, 60, 100, 200, 400]` with one uniform budget
across the current failure, semantic-contact maximum, minimum full-surface SDF,
maximum interaction-energy, frames `0/29/59`, RH/LH, and a successful
pre-contact case. The auditable result and deterministic-repeat records are
kept in `.local/reports/stage9_1/maxiter_benchmark.json`; the selected budget
and final v2 profile hash are copied into the development log and Stage 10
manifest. Solver and termination behavior remain implementation assumptions
because the paper does not disclose them.
The measured v1 profile hash is `6affff2fdb425a0402f643c291c0b8904d4dbec6c5b69a5006cf9829dcc220aa`;
the independent v2 profile hash is
`c42c21d894c54d07b1d30943b5a3338b13628bf0429ab203b5540cf934d09b7c`. The fixed
35-record grid selects uniform `maxiter=100`. The Stage 9.2 full 60-frame
contact-rich artifact and fresh/resumed deterministic comparison are now
recorded in `.local/reports/stage9_performance/`; the reference-runtime minimum
gate passes while the preferred single-frame gate remains unmet.

## Stage 9.2 execution layer

The performance and recoverability implementation is documented in
[`REFINEMENT_PERFORMANCE.md`](REFINEMENT_PERFORMANCE.md) and
[`REFINEMENT_CHECKPOINT_AND_RESUME.md`](REFINEMENT_CHECKPOINT_AND_RESUME.md).
It owns callback caching, persistent resources, batched point Jacobians,
scheduled independent full-surface audits, and atomic frame checkpoints. These
are engineering mechanisms only; Eq. (8)-(9), profiles, weights, sample count,
and strict acceptance are unchanged. Runtime-gate and deterministic-repeat
claims require reports under `.local/reports/stage9_performance/`.

## Stage 9.3.3 shadow-equivalence boundary

Stage 9.3.3 replays the frozen formal profile only in isolated diagnostic
outputs. It first calibrates `toporetarget.shadow_equivalence.v1` from three
independent official-profile repeats, with predeclared float64 floors, a `20x`
repeat-noise multiplier, and hard caps. A context, status, QuerySet identity,
or numerical-equivalence failure stops all other profiles. The six optional
shadow profiles do not alter Eq. (8)-(9), solver/execution YAML, paper weights,
or accepted Stage 9.2/Stage 10 artifacts; projections are state diagnostics,
not accepted trajectories. See
[`SHADOW_EQUIVALENCE_AND_LONG_FINGER_ABLATION.md`](SHADOW_EQUIVALENCE_AND_LONG_FINGER_ABLATION.md).
