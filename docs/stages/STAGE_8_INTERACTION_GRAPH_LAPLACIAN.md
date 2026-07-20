# Stage 8 — interaction graph and Laplacian coordinates

Status: `implemented_with_assumptions` for the bounded RH/LH acceptance window.

## Scope

Stage 8 closes the paper's Eq. (3)-(7) graph/deformation term only:

- 21 canonical source hand vertices plus 50 audited Stage 6 object samples;
- source-only SciPy/Qhull Delaunay tetrahedralization;
- complete unique tetrahedron-edge extraction;
- source-derived directed distance weights with `kappa=30`;
- source and robot weighted Laplacian coordinates on the same cached graph;
- exact Eq. (7) mean squared residual divided by 71;
- bounded qpos Jacobian and base perturbation diagnostics at frozen warm starts.

Eq. (8)-(9), optimization, slack, SDF, collision penalties, RL, and full
dataset evaluation are explicitly out of scope.

## Acceptance artifacts

The real bounded `s7/cubemedium_inspect_1` window uses 60 frames for both hands:

```text
.local/cache/retarget/interaction_graph/
.local/cache/retarget/interaction_evaluation/
.local/reports/stage8/
```

The reports include input audit, graph/evaluation validation, identity oracle,
qpos Jacobian validation, topology-over-time, object-scale diagnostics,
performance, source-integrity hashes, and first/middle/last-frame visualizations.
All are ignored local derived outputs; no external dataset/model is copied into
the repository.

The bounded interactive viewer smoke test covers slider, previous/next, play/pause,
source/robot visibility, three edge-category toggles, Laplacian/residual/contribution
toggles, and timer cleanup while reusing the saved graph.

## Explicit numerical assumption

The strict profile uses `Qbb Qc Qz Q12` and no random jitter. Raw metre-scale
coordinates exposed zero-volume Qhull edge cases on this audited sequence, so
the implementation passes a deterministic centroid-translated,
bounding-box-diagonal-normalized copy to Qhull. This is a numerical conditioning
transform only: source vertices, volumes, distances, weights, and artifact
hash inputs remain in metres. The diagnostic jitter profile is never used for
the acceptance artifacts.

## Boundary invariant

The graph builder does not import a robot model or Stage 7 warm start. The
evaluation path loads the saved graph and warm start, reuses source topology and
weights, and reports `robot_delaunay_invocation_count=0`,
`optimization_performed=false`, `sdf_accessed=false`, and
`collision_surface_accessed=false`. Any change of source geometry, object scale,
or sample identity requires a new graph artifact.
