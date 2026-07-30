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
