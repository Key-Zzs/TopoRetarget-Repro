# Final Refinement Performance Repair

Stage-12 batch final work is currently paused. The repair isolates all new
outputs under `.local/experiments/final_refinement_perf_v1/` and reports under
`.local/reports/final_refinement_perf/`; original checkpoints and formal
artifacts are not overwrite targets.

The incident has two measured causes: the paused legacy workers each exposed
about 79--80 threads and collectively oversubscribed the host, while the
non-smooth finite-difference constraint Jacobian recomputed a complete active
QuerySet for every row/column perturbation. The repair batches those exact
probes into one SDF backend query without changing the central-difference
definition. New work starts with one worker and one BLAS/Torch thread. Legacy
workers without a validated atomic checkpoint stay stopped via their exact PGID
rather than being killed.

The performance candidate is an engineering execution profile, not a paper
claim or author-exact implementation. Its numerical-equivalence gate covers
SDF/closest-point/normal values, collision points/Jacobians, objectives,
constraints, active-set behavior, strict acceptance, and final independent
audits before any future resume transition may be recommended.

## P2 analytic-SDF qualification

`wuji_continuous_sequential_fast_exact_v2` keeps the v1 mathematical contract
but uses chain-rule spatial SDF gradients, certified Lipschitz sign reuse, and
an exact object-local BVH. The fixed frames 0/12/29/45/59 completed with strict
acceptance and a 9.129 s median (11.470 s p95); see the P2 report under
`.local/reports/final_refinement_p2/reports/`. This is not a Stage-12 default
or a resume action: explicit operator approval remains required.
