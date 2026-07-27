# Signed distance and collision queries

Stage 6 provides geometric query foundations only. It does not construct the paper's final `Q_t`,
Delaunay graph, Laplacian coordinates, constraints, slack variables, or optimization variables.

## Query contract

`SignedDistanceQueryResult` returns signed and unsigned distance, exact closest point, source face
index, closest barycentric coordinate, outward normal, inside/on-surface flags, validity, sign
confidence, sign method, backend ID, and mesh hash. The repository convention is always

```text
signed_distance > 0  outside
signed_distance = 0  on the surface
signed_distance < 0  inside
```

The reference backend evaluates the analytic point-to-triangle closest-point regions in chunks.
It never substitutes nearest vertices or sampled object points for mesh distance. Optional `rtree`,
`pyembree`, and `embreex` are not required; their availability is reported and the reference path
remains the correct fallback.

## Sign modes

* `strict` requires a watertight, orientable, consistently wound, non-degenerate mesh. It fails
  with the mesh audit when that contract is not met.
* `winding` computes a chunked generalized winding number from triangle solid angles. The threshold
  and ambiguity confidence are recorded; open meshes are never upgraded to reliable signs.
* `unsigned_only` returns closest points and unsigned distance, with signed distance as `NaN`,
  `inside=None`, and `sign_valid=False`. It is suitable for visual/closest-point diagnostics only.

Source winding is not repaired. For a consistently reversed closed mesh, the derived normal view is
oriented outward from the signed-volume convention while the source bytes remain unchanged. A
closest point on an edge or vertex is marked `non_smooth`; low-confidence signs invalidate the local
linearization. The local first-order data is `phi`, closest point, outward normal, and validity; no
robot-q Jacobian or SQP implementation is included.

Scene queries reuse the existing SE(3) helpers: points are transformed by `(T^S_O)^-1`, queried in
`O`, and closest points/normals are transformed back (normals rotate only). Rigid transforms do not
change distance.

## Robot collision surfaces

`RobotHandModel.collision_geometry_instances()` is the only robot input. Mesh, sphere, box, and
cylinder collision geometry are sampled in their local frames, then transformed by the existing
URDF/FK chain into robot base and scene frames. Visual geometry is never a silent fallback, and
fixed visual-only tip spheres are reported as missing collision coverage. The explicit engineering
profile is `engineering_collision_32_per_geometry`; the paper does not publish this count.

`query_robot_surface_against_object` is pointwise and returns each sample's distance, closest point,
normal, sign confidence, link/geometry/sample IDs, and `max(-d, 0)` penetration depth only when the
sign is valid. It has explicit `final_query_set=false` and `optimization=false` provenance.

```bash
toporetarget geometry validate-sdf --shape sphere --report .local/reports/stage6/sdf_sphere_validation.json
toporetarget geometry sdf-query --canonical "$GRAB_CACHE" --object-id primary --frame 0 \
  --points .local/reports/stage6/probe_points.npy --points-frame scene --sign-mode strict
toporetarget geometry sample-robot --robot artimano_rh --pose neutral \
  --profile engineering_collision_32_per_geometry --output .local/cache/geometry/robot_surface/rh.npz
toporetarget geometry probe-collision --robot-samples .local/cache/geometry/robot_surface/rh.npz \
  --object-shape cube --report .local/reports/stage6/synthetic_collision_probe.json
```

For non-watertight GRAB meshes, use `winding` for confidence diagnostics or `unsigned_only` for
closest-point diagnostics. Do not treat either as a reliable strict penetration sign without a
mesh audit supporting it.

Stage 9 uses this strict positive-outside reference contract for its final constrained
interaction-preserving refinement and independent full-surface audit. Its QuerySet/slack policy is
documented in [COLLISION_QUERY_SET_AND_SLACK.md](COLLISION_QUERY_SET_AND_SLACK.md); Stage 6
sampling and SDF inputs remain read-only.

## Open meshes in the quality lane

The quality lane uses the [hybrid open-object contract](HYBRID_SIGNED_DISTANCE_FOR_OPEN_OBJECTS.md).
The original mesh supplies closest points, original face IDs, normals, and the
unsigned magnitude; a deterministic watertight proxy supplies only the sign.
Results expose `proxy_closest_face_indices`,
`proxy_closest_is_synthetic_patch`, `original_boundary_distance`, and
`near_original_boundary`. Source-contact and active-QuerySet boundary conflicts
fail closed as `SIGN_PROXY_CONTACT_REGION_CONFLICT`.
