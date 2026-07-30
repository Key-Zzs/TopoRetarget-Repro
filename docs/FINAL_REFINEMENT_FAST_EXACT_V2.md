# Final Refinement Fast-Exact v2

v1 is frozen as the immutable optimizer-coordinate-FD performance reference. v2 is a
non-default, CPU float64 candidate with object-local exact BVH, spatial-gradient chain
rule, ambiguity-only 3D FD, and certified sign reuse. Eq. (1)--(9), collision-sample
identity, active-set behavior, solver limits, retry cascade, and independent final audit
are unchanged. See `.local/reports/final_refinement_p2/reports/p2_summary.json` for
real qualification measurements. Stage-12 remains paused pending user approval.
