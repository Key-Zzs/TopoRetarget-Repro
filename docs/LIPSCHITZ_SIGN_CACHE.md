# Lipschitz-Certified Sign Cache

The v2 cache stores only exact-sign provenance by stable QuerySet sample ID. Closest
points and unsigned distances are still queried exactly. A sign is reused only if
`||x_new-x_old|| + sign_safety_margin + surface_epsilon < abs(phi_old)`, with matching
mesh/profile hashes and a reliable prior exact sign. This is the 1-Lipschitz certificate
that excludes a surface crossing. Arbitrary batches and spatial-FD probes do not update
the cache; misses use exact generalized winding/proxy sign evaluation.
