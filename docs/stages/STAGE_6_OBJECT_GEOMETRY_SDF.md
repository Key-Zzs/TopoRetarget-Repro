# Stage 6 — object geometry, deterministic sampling, and signed distance

## Objective and scope

Stage 6 supplies the geometric inputs required by later interaction/collision work while preserving
the boundary of the paper reproduction. It audits object meshes, generates exactly 50 deterministic
object-local references, supports closest-point and signed-distance queries, samples existing robot
collision geometry, and provides pointwise collision probes. It does not start Stage 7 and does not
implement relative bone directions, warm starts, Delaunay, interaction graphs, Laplacian losses,
final optimization, `Q_t`, slack, or PPO.

## Data flow

```text
canonical object-local mesh
  -> read-only mesh audit
  -> area-weighted face selection
  -> square-root barycentric sampling
  -> fixed 50 local anchors
  -> T^S_O scene transform per frame
  -> triangle closest point + signed distance
  -> closest point / normal / sign confidence
```

```text
Stage 4 collision geometry
  -> link-local surface anchors
  -> existing FK
  -> robot base/scene points
  -> object SDF point probes
```

## Files and validation

Core implementation lives in `src/toporetarget/geometry/mesh_audit.py`,
`surface_sampling.py`, `surface_artifacts.py`, `object_geometry.py`, `robot_surface.py`,
`collision_queries.py`, `reports.py`, `visualization.py`, and `signed_distance/`. CLI integration is
`src/toporetarget/cli/geometry.py`. Profiles are in `configs/geometry/`; tests are in the Stage 6
unit/integration/local-data/local-asset files. The detailed assumptions are registered in
`docs/ASSUMPTIONS.md` and the machine-readable fidelity entries in `docs/PAPER_FIDELITY.yaml`.

The bounded real acceptance uses the existing Stage 5 canonical cache
`s7/cubemedium_inspect_1`, frames `[0,60)`, and the imported local RH/LH Arti-MANO assets. No full
GRAB conversion is performed and no raw mesh, canonical cache, MANO, Arti-MANO, or source NPZ is
modified.

## Definition of done

The Stage 6 report is accepted when the synthetic audit/sampling/SDF/open-mesh/robot-probe tests,
paper-fidelity checker, default suite, bounded real object validation, RH/LH collision sampling,
source-integrity check, and required visual artifacts pass. A real mesh that cannot provide a strict
sign is reported as `real_mesh_sign_limitation`; it is never silently repaired or relabeled.

Stage 7 is documented separately in
[`STAGE_7_BONE_DIRECTION_WARM_START.md`](STAGE_7_BONE_DIRECTION_WARM_START.md). Its bounded
relative-bone-direction warm-start path does not consume this Stage 6 object geometry or SDF
output; Stage 8 and later interaction/refinement work remain not started.
