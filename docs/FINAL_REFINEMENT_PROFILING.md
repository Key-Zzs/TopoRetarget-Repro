# Final Refinement Profiling

`toporetarget retarget profile-refinement` profiles selected real frames into a
new diagnostic root. It records scoped timers, callback counts, exact-x cache
statistics, cProfile output, and a bounded profiler-availability report without
overwriting a formal final artifact.

Frame solve time begins at `refine_frame` and ends after strict acceptance,
independent audit, and checkpoint commit. Dataset loading, canonical conversion,
and HTML generation are reported separately as cold-start overhead. The Stage-12
performance report must publish measured five-frame values only; missing or
timed-out reference values are labeled `N/A`, never estimated.

The P2 analytic-SDF profile additionally records analytic versus ambiguity-only
spatial-FD rows, sign-cache provenance, exact-winding counts, BVH traversal
statistics, and per-frame cProfile output. Its five-frame qualification keeps
the legacy baseline immutable and writes all generated evidence to `.local/`.
