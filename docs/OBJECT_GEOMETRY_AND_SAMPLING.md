# Object geometry and deterministic surface sampling

Stage 6 keeps every object mesh in its canonical object-local frame `O`. The existing
`MeshDefinition` and `RigidObjectTrack` remain the single source of truth: `mesh.vertices_local`
are object-frame vertices and `pose_scene` is `T^S_O`.

## Stage 8 consumer boundary

Stage 8 consumes the audited 50-point `paper_strict_area_uniform` artifact as fixed
face+barycentric anchors. It transforms those anchors by the canonical object pose into
scene frame `S`, concatenates them after the 21 hand points, and never resamples them per
frame. Sampling, mesh, and topology hashes are checked before graph construction. SDF and
collision geometry remain Stage 6/9 interfaces and are not interaction-loss inputs.

## Audit

`toporetarget geometry inspect-mesh` performs a read-only audit. It reports finite vertices,
triangle/index validity, zero and near-zero faces, duplicate and unreferenced data, boundary and
non-manifold edges, components, watertightness, winding/orientability, signed volume, Euler number,
bounds, area, center of mass, array/topology hashes, provenance, and sign reliability. The audit
never deletes faces, merges vertices, repairs winding, fills holes, scales data, or overwrites a
source mesh. Degenerate faces can be excluded from derived sampling/query views only, with the
count recorded.

## Paper count and engineering profile

`paper_strict_area_uniform` resolves `count` from
`configs/paper/retarget.yaml:num_object_surface_samples`, so the paper value remains defined in
one place and resolves to exactly 50. The paper does not publish a sampler, seed, temporal reuse
schedule, or normal mode. The profile therefore has status `implementation_assumption` and records
assumptions `A_OBJECT_SAMPLING_METHOD_001`, `A_OBJECT_SAMPLING_SEED_001`,
`A_OBJECT_SAMPLE_TEMPORAL_REUSE_001`, and `A_SURFACE_NORMAL_MODE_001`.

The implementation uses an explicit `numpy.random.Generator(numpy.random.PCG64(20260720))`,
area-weighted triangle selection, and square-root barycentric sampling. Each sample stores its
face index and barycentric coordinate as well as its reconstructed point and diagnostic face
normal. Face+barycentric anchors are retained because they make the point exactly reconstructible
after scale changes and preserve sample identity; storing only coordinates would lose that
provenance. FPS is not used because it is not required by the paper and would produce a different
engineering profile.

Sampling happens once in `O`. For each frame, the same anchors are transformed by `T^S_O`; there is
no temporal resampling. The resulting identity is stable across first/middle/last frames. Uniform
and non-uniform scale are derived views: vertices are scaled, points are reconstructed from the
anchors, and normals are recomputed. Raw/canonical meshes are never changed.

## Artifacts and commands

The explicit artifact is an `.npz` containing face indices, barycentric coordinates, local points,
normals, masks, profile metadata, mesh/topology/profile hashes, scale, and provenance. It is
disposable under `.local/cache/geometry/object_surface/`; the key includes mesh, topology, profile,
scale, and code-version inputs. Loading rejects hash mismatches and does not copy the source mesh.

```bash
toporetarget geometry inspect-mesh --canonical "$GRAB_CACHE" --object-id primary \
  --json .local/reports/stage6/grab_object_mesh_audit.json
toporetarget geometry sample-object --canonical "$GRAB_CACHE" --object-id primary \
  --profile paper_strict_area_uniform --output .local/cache/geometry/object_surface/object.npz \
  --report .local/reports/stage6/object_samples.json
toporetarget geometry validate-samples --canonical "$GRAB_CACHE" --object-id primary \
  --samples .local/cache/geometry/object_surface/object.npz \
  --report .local/reports/stage6/object_samples_validation.json
```

The object viewer reuses the repository Matplotlib conventions and can show mesh, 50 samples,
normals, IDs, object/scene frames, and a selected frame. It is a diagnostic viewer, not a new
canonical-data representation.
