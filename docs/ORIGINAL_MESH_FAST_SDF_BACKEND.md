# Original-Mesh Fast SDF Backend

`OriginalMeshSignedGridSDFBackend` is an engineering acceleration for Stage 9,
not a paper claim.  It is selected only after the frozen temporal-sync replay
shows a fast/reference geometry mismatch.  It never replaces the independent
triangle-winding final audit.

## Geometry contract

- Input is the original object-local mesh and must pass the strict watertight,
  orientable, consistently wound mesh audit.
- The source mesh is never repaired, replaced by a convex hull, or modified
  with an object-specific parameter.
- Grid-node signed distances come from `ReferenceSignedDistanceBackend` with
  strict generalized winding and exact closest-triangle distance.  The sign is
  positive outside.
- Grid profile IDs are `original_mesh_signed_grid_192_v1` and
  `original_mesh_signed_grid_256_v1`.  The latter is evaluated only if the
  former fails its predeclared accuracy gate.

## Construction and caching

The object-local padded bounds use the fixed rule:

```
padding_m = max(0.02, 0.25 * object_bbox_diagonal)
```

The selected resolution is the node count on the longest padded axis; other
axes preserve that voxel spacing.  A cache key includes the raw mesh hash,
profile ID, padding, origin, shape, and voxel size.  Each completed z-slab is
stored atomically in a resumable partial grid.  A completed grid stores its
signed values, spatial gradients, and construction metadata under the current
worktree's `.local` experiment tree.

## Query contract

Queries accept and return float64 values.  Signed distance and spatial
gradient use trilinear interpolation.  Outside the padded grid, the backend
does not clip to a negative value: a point that is provably outside the
original mesh bounding box receives its exact closest-triangle unsigned
distance with a positive sign.  This is a valid positive exterior lower bound
and does not run winding.  Any other out-of-grid case fails closed.  Out-of-grid
queries are counted and excluded from analytic gradient validity.

## Selection gate

Both frozen Stress1 and Stress2 must pass against the strict reference on
surface-near, penetration-active, outside, and replay collision samples:

- sign agreement at least 0.99;
- reference `>1 mm` recall at least 0.95 when applicable;
- penetration-depth correlation at least 0.95;
- gradient cosine at least 0.90;
- p95 absolute signed-distance error at most 0.25 mm;
- active-region maximum error at most 0.75 mm;
- finite, rigid-transform-equivariant, deterministic results.

Only reference consistency, gradients, runtime, and memory determine backend
selection.  E0/S1 quality results do not.  If 192 fails, 256 is tested; if 256
also fails, the task terminates with `ORIGINAL_MESH_FAST_BACKEND_REJECTED`.

When accepted, both E0-grid (`lambda_sdf=0`) and S1-grid
(`lambda_sdf=0.1`, 1 mm dead-zone profile) use the same selected inner backend;
final artifact validation remains `reference_winding_v1`.
