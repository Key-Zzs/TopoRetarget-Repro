# Stage 8 interaction graph

Stage 8 implements the graph portion of the paper's Eq. (3)-(6) with a strict
source/robot boundary. The graph has 71 vertices in scene frame `S`:

```text
0..20   canonical source or robot MediaPipe-21 hand points
21..70  the same 50 posed Stage 6 object samples, in fixed face+barycentric order
```

For every selected source frame, `src/toporetarget/retarget/delaunay.py` runs
one non-incremental `scipy.spatial.Delaunay` call using the tracked profile
`configs/retarget/interaction/strict_scipy_qhull_v1.yaml`. The strict profile
records `Qbb Qc Qz Q12`, rejects `QJ`, and uses only centroid translation plus
uniform bounding-box-diagonal scaling for Qhull conditioning. The vertices,
volumes, and Eq. (5) distances remain in the original metre scene frame.

All six edges of each returned tetrahedron are extracted, globally deduplicated,
sorted, and retained. Directed weights are computed once from source squared
distances with `kappa=30` loaded from `configs/paper/retarget.yaml`, then
row-normalized. There is no robot-side Delaunay, edge filtering, jitter, point
merging, or object resampling. Invalid duplicate, near-duplicate, coplanar,
zero-volume, isolated, or no-hand-object-connectivity frames fail loudly.

The graph artifact is a Zarr group with root attributes and schema
`toporetarget.interaction_graph.v1`. Ragged simplex/edge/adjacency arrays,
source Laplacians, provenance hashes, profile hashes, frame statistics, and
per-frame graph hashes are stored. `inspect-interaction-graph` emits a frame JSON
and directed-edge CSV for debugging; `topology_over_time` records edge Jaccard
and graph-hash changes without treating topology changes as errors.

The interactive viewer reuses the saved arrays and exposes frame slider,
previous/next, play/pause, source/robot visibility, hand-hand/hand-object/object-object
edge toggles, Laplacian/residual/contribution toggles, and close/timer cleanup. Frame
changes never invoke Delaunay.

## Commands

```bash
toporetarget retarget audit-interaction-inputs \
  --right-canonical RH.zarr --left-canonical LH.zarr \
  --right-warm-start RH_warm.zarr --left-warm-start LH_warm.zarr \
  --object-samples cubemedium_samples.npz \
  --report .local/reports/stage8/input_audit.json

toporetarget retarget build-interaction-graph \
  --canonical RH.zarr --hand right --object-samples cubemedium_samples.npz \
  --output .local/cache/retarget/interaction_graph/right.zarr \
  --report .local/reports/stage8/rh_graph_build.json
```

Stage 8 evaluation loads this artifact and the Stage 7 warm-start artifact,
reuses its exact connectivity and weights, and never mutates qpos/base. Eq. (7)
is evaluated as the mean squared Eq. (6) residual over 71 vertices. Constrained
optimization, slack, SDF/collision penalties, and Eq. (8)-(9) are implemented in
the bounded Stage 9 workflow documented in
[stages/STAGE_9_FINAL_CONSTRAINED_REFINEMENT.md](stages/STAGE_9_FINAL_CONSTRAINED_REFINEMENT.md).
