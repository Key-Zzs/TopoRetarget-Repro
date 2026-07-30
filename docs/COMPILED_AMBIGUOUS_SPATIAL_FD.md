# Compiled Ambiguous Spatial FD

The v3 experiment keeps v2 ambiguity routing unchanged. Only rows classified
as `SPATIAL_FD_REQUIRED` enter the compiled closest-point query. A batch is
materialized as contiguous `[N, 6, 3]` probes for `+/-x`, `+/-y`, and `+/-z`.

Closest points and unsigned distances come from the compiled source-mesh BVH.
The exact hybrid proxy/generalized-winding sign backend still classifies every
probe, so a base-point sign is never copied across a surface crossing. FD
probes never update the formal Lipschitz sign cache.
