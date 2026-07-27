# Derived watertight sign proxy

This repository uses `hybrid_original_distance_proxy_sign_v1` for an audited
open-object sign policy. It is a deterministic, object-independent engineering
profile and is not a paper-specified geometry method.

The source mesh is read-only. `Candidate 0` is identity for a valid watertight
mesh. An open mesh is copied into a derived workspace and tried in this order:

1. `Candidate 1`, deterministic local topological repair: remove duplicate and
   degenerate faces, merge exact duplicate vertices, clean unreferenced data,
   orient faces, and fill closed boundary loops with synthetic triangles.
2. `Candidate 2`, fixed 256-longest-axis voxel occupancy, at most one fixed
   voxel closing, and marching cubes, only if Candidate 1 fails its strict
   gates.

Convex hulls, manual object-specific edits, replacement meshes, trajectory-
dependent repair, smoothing, scaling, and contact-dependent parameters are not
accepted proxy candidates. Every selected candidate records its method, source
and proxy hashes, source-face mapping, synthetic patch IDs, boundary loops,
surface-deviation samples, and policy hash.

The fixed acceptance gates are watertightness, consistent winding, orientability,
zero boundary/non-manifold/degenerate faces, positive finite signed volume, and:

```text
p95 <= max(0.001 m, 0.005 * bbox_diagonal)
max  <= max(0.003 m, 0.015 * bbox_diagonal)
bbox_extent_relative_error <= 0.01
synthetic_patch_area_ratio <= 0.05  # Candidate 1
```

Artifacts are written below:

```text
.local/experiments/grab_artimano_quality_v1/geometry/
  source_mesh_audit.json
  source_mesh_audit.csv
  geometry_manifest.json
  <source_mesh_hash>/proxy_manifest.json
  <source_mesh_hash>/source_mesh.npz
  <source_mesh_hash>/source_distance_mesh.npz
  <source_mesh_hash>/proxy_mesh.npz
```

The proxy is used only for inside/outside sign classification. Visualization,
object samples, closest points, unsigned magnitudes, contact-position targets,
and provenance remain tied to the original mesh. A failed candidate is reported
as `DERIVED_SDF_PROXY_FAILED`; it must not silently fall back to unsigned-only
Stage 9 behavior.
