# Stage 10 trajectory visualization and reference export

Visualization is manifest-driven and artifact-only. It delegates rendering to the
existing Stage 9 viewer and never invokes the refinement solver. The available
layers are source hand, warm start, final hand, object, interaction edges,
collision samples, adaptive query membership, penetration markers, and slack.
The workflow wrapper currently exposes the Stage 9 scene view; `--view scene` is
accepted for an explicit, stable command contract.

Render one named frame:

```bash
toporetarget workflow visualize \
  --run .local/runs/stage10/<run>/manifest.json \
  --view scene --frame 29 --output .local/runs/stage10/<run>/review/frame029.png \
  --report .local/runs/stage10/<run>/review/frame029.json
```

The review bundle also records first/middle/last and metric-worst frames plus a
replayable command. Interactive inspection is explicit:

```bash
toporetarget workflow visualize \
  --run .local/runs/stage10/<run>/manifest.json \
  --interactive --view scene \
  --show-source-hand --show-warm-start --show-final --show-object \
  --show-interaction-edges --show-collision-samples --show-query-set \
  --show-penetrations --show-slack
```

For a headless animation, the wrapper renders the requested local frame range
through the same Stage 9 renderer and assembles a Pillow GIF; it does not call a
solver:

```bash
toporetarget workflow visualize \
  --run .local/runs/stage10/<run>/manifest.json \
  --start-frame 0 --end-frame 60 --display-stride 1 \
  --output .local/runs/stage10/<run>/review/trajectory.gif
```

In the interactive viewer, drag the frame slider to inspect a frame, use
Space to play or pause, and Left/Right to step by one frame. The layer switches
control source/warm/final hands, object, graph, collision samples, query set,
penetrations, and slack. Review-frame navigation is recorded in
`review/review_frames.json`; the generated `visualize_command.txt` is the exact
artifact-resolved launch command.

Reference export is separate from visualization and performs no solver call:

```bash
toporetarget workflow export-reference \
  --run .local/runs/stage10/<run>/manifest.json \
  --format zarr
```

The exported `toporetarget.robot_reference.v1` contains timestamps, native frame
indices, qpos, scene base poses, robot keypoints/link poses, object poses, and
content/provenance hashes. It is an offline reference artifact, not a hardware
command stream.

## Interaction-mesh HTML viewer

`visualize-mesh` writes a self-contained HTML page from the accepted manifest. It
loads the canonical source mesh, warm-start and final robot visual meshes, the
frozen Stage 8 graph/evaluation artifacts, and the final keypoints. It does not
invoke a solver, rebuild Delaunay connectivity, recompute Stage 8 weights, or
write any input artifact.

```bash
toporetarget workflow visualize-mesh \
  --run .local/runs/stage10_reference_runtime/s1__airplane_lift__right__artimano_rh__f000240_f000300/manifest.json \
  --mode combined \
  --interactive
```

The page supports five initial modes, all switchable without regenerating data:

| Mode | Display | Default purpose |
| --- | --- | --- |
| `mesh` | source, warm-start, final meshes and object context | inspect pose/mesh alignment |
| `full-graph` | the same meshes plus all graph states and HH/HO/OO edges | inspect frozen connectivity |
| `figure4-style` | the same meshes plus graph states with hand-object edges emphasized | inspect interaction structure |
| `laplacian-diagnostic` | the same meshes plus graph and residual scalar/vector diagnostics | locate deformation mismatch |
| `combined` | mesh layers, graph, and residual diagnostics | side-by-side review |

The graph contains the 21 hand vertices followed by 50 Stage 6 object samples.
The source/warm/final states share the exact saved connectivity and object-point
identity. Directed Stage 8 weights are preserved; only the display line weight is
`w_vis(i,j) = (w_ij + w_ji) / 2`. Final residuals are read-only diagnostics using
the frozen directed graph weights. The sidebar can filter by category, threshold,
top-k, hand-object-only, residual target/scope, scalar/vector display, and labels.

The viewer reports `max`, mean, hand mean, object mean, and top residual vertices.
These are diagnostic values, not an acceptance gate. Judge retarget quality with
the Stage 9/10 numeric reports as well: interaction objective/residual, collision
and penetration, temporal continuity, joint/base limits, solver acceptance, and
the required artifact/provenance checks. See
[`INTERACTION_MESH_VISUALIZATION.md`](INTERACTION_MESH_VISUALIZATION.md) for the
full interpretation and recommended review procedure.

## Stage 9.3 contact audit viewer

The contact-retention audit has a separate self-contained HTML review. It reads
the accepted manifest-resolved artifacts and does not call the solver:

```bash
toporetarget workflow audit-contact-retention \
  --run .local/runs/stage10_reference_runtime/<run>/manifest.json \
  --output-dir .local/runs/stage9_3_contact_audit/<run> \
  --surface-samples 8192 --thresholds-mm 1,2,3,5,8,10 --html --force
```

The page separates source, warm-start, final, object, visual surface, collision
geometry, QuerySet, anchors, and nearest object segments. Frame, threshold, and
link/region controls are review aids; per-frame/per-link CSV and JSON reports
remain authoritative. Object points are transformed from object-local into the
scene frame for display. Source contact and semantic-anchor retention are
diagnostic proxies rather than ground-truth labels, and the warm-to-final
interpolation is not an optimizer trajectory.
