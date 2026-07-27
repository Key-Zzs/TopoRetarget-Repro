# Hybrid signed distance for open objects

The formal backend ID is
`hybrid_original_distance_proxy_sign_v1`, schema
`toporetarget.derived_sdf_proxy.v1`.

For a query point `x`, the backend computes the closest point, original face,
surface normal, and unsigned distance `d_original(x)` on the cleaned original
surface. It independently computes `inside_proxy(x)` and sign confidence on a
strictly watertight derived proxy, then returns:

```text
phi(x) = -d_original(x)  if inside_proxy(x)
         +d_original(x)  otherwise
```

Distance to a synthetic patch is never used as the formal magnitude. Results
also retain the proxy closest face, synthetic-patch flag, original-boundary
distance, fixed boundary-exclusion flag, profile/cache hashes, and geometry
provenance.

The fixed boundary-exclusion radius is
`max(0.002 m, 0.01 * bbox_diagonal)`. Source semantic contact samples and the
Stage 9 active QuerySet are audited separately. A nonzero source-contact or
active-QuerySet conflict is a hard
`SIGN_PROXY_CONTACT_REGION_CONFLICT`; it is not suppressed by selecting a
different frame, changing the margin, changing the trajectory, or skipping a
sample.

For already valid meshes, identity validation compares the strict reference
backend, hybrid backend, and independent validator. The required identity
bound is zero sign mismatch and maximum absolute signed-distance difference no
greater than `1e-10 m`. Open-object validation additionally checks magnitude
identity, finite proxy signs, rigid-transform equivariance, object-local/scene
round trip, finite-difference behavior, and deterministic CPU float64 output.

This policy is paper-unspecified geometry engineering. It does not change the
paper objective, solver profiles, interaction-graph object samples, raw mesh,
visualization mesh, or old Stage 10 artifacts.
