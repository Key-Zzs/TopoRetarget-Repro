# Interaction-mesh HTML visualization

The Stage 10 `visualize-mesh` command produces one self-contained HTML page. It
combines the existing source/warm/final hand-mesh view with the accepted Stage 8
interaction graph and a read-only Laplacian residual diagnostic.

```bash
toporetarget workflow visualize-mesh \
  --run .local/runs/stage10_reference_runtime/s1__airplane_lift__right__artimano_rh__f000240_f000300/manifest.json \
  --mode combined \
  --output .local/runs/stage10_reference_runtime/s1__airplane_lift__right__artimano_rh__f000240_f000300/review/trajectory_combined.html \
  --interactive
```

`--interactive` opens the generated file in the default browser. The HTML is
interactive even when the flag is omitted. A generated page contains all five
modes and the mode selector changes only the drawing state:

| Mode | What is shown | Initial defaults |
| --- | --- | --- |
| `mesh` | source MANO, warm-start Arti-MANO, final Arti-MANO visual meshes | all mesh layers and object context |
| `full-graph` | the same meshes plus source/warm/final graph states and all edge categories | HH, HO, and OO enabled |
| `figure4-style` | the same meshes plus graph states with interaction structure emphasized | HO-only; HH and OO hidden |
| `laplacian-diagnostic` | the same meshes plus graph and residual scalar heat/vector arrows | warm residual target |
| `combined` | mesh, graph, object context, and residual diagnostics | all relevant layers |

## Meaning of the extra points and lines

The filled triangles are the three hand meshes. The additional points and lines
are diagnostics, not another mesh wireframe:

- Gray points are the `object context` point cloud, sampled from the object mesh
  for scene context; the HTML payload caps this display at 1200 points.
- Blue, orange, and green graph points are the source, warm-start, and final
  graph states. Each state has 21 hand keypoints followed by the same 50 object
  sample points, for 71 graph vertices total.
- Graph lines connect graph vertices. They are categorized as hand-hand,
  hand-object, or object-object edges from the saved Stage 8 topology. The
  source, warm-start, and final states use the same connectivity but different
  hand-keypoint coordinates.
- Residual-colored larger points show residual magnitude: blue is smaller and
  red is larger after per-frame normalization. Red arrows show the residual
  vector direction; the arrow is visually scaled for readability.

In graph modes the graph and residual are drawn over the scene, and the mesh
layers remain enabled by default. The mesh checkboxes can hide any of the three
meshes independently. Graph state colors are blue/source, orange/warm-start,
and green/final; they should not be confused with the residual heat color.

## Sidebar controls

| Control | Meaning |
| --- | --- |
| `Visualization mode` | Select `mesh`, `full-graph`, `figure4-style`, `laplacian-diagnostic`, or `combined`. Every mode keeps the mesh layers available. |
| `Frame` / `Play` | Select a local frame or play the trajectory. The label also shows the native source frame. |
| `Mesh layers: source/warm/final` | Toggle the blue source MANO, orange warm-start, and green final robot visual meshes. |
| `Object context` | Toggle the gray object surface point cloud. |
| `Graph states: source/warm/final` | Toggle the three graph states independently. |
| `Labels` | Show semantic names or object sample IDs beside graph vertices. |
| `hand-hand`, `hand-object`, `object-object` | Enable or hide each graph edge category. |
| `hand-object only` | Additional filter that keeps only hand-object edges; it is enabled by default in `figure4-style`. |
| `weight mode` | `none` uses normal lines; `opacity`, `width`, and `color` map the display-only edge weight to transparency, thickness, or hue. |
| `edge threshold` | Hide edges whose display weight is below the selected value. |
| `top-k edges` | Keep only the strongest K currently eligible edges; `0` means all. |
| `residual target` | Select the saved warm-start residual or the read-only final residual. |
| `residual display` | Show residual magnitude as scalar color, direction as vectors, or both. |
| `residual scope` | Restrict residual display to all vertices, hand vertices, or object vertices. |
| `residual threshold` | Hide residual vectors/points below the selected magnitude. |
| `top-k residual vertices` | Show only the K largest residual vertices; `0` means all. |
| `Frame metrics` | Show the current refinement, collision, acceptance, and solver-related metrics from the artifact. |
| `Provenance` | Show the frozen graph hash and the directed-weight display convention. |

The edge threshold and top-k filters affect only what is drawn. They do not
remove edges from the artifact or change the graph used by the retargeting
algorithm. Likewise, residual thresholds and scopes are visualization filters,
not acceptance thresholds.

## Data and provenance contract

The page reads the paths in the manifest for canonical, warm-start, graph,
evaluation, and final artifacts. It validates equal frame counts and verifies
that `graph.source_vertices[:, 21:]` is exactly the object-sample identity used
by the Stage 8 evaluation. Final graph hand vertices come from the final
trajectory; the 50 object vertices remain the frozen Stage 8 samples.

The graph remains the complete saved Stage 8 Delaunay edge set. The sidebar
filters only drawing: hand-hand, hand-object, object-object, threshold, top-k,
and hand-object-only. The persisted directed weights are never changed. For a
line between an undirected edge's endpoints, the viewer uses the display-only
mean `(w_ij + w_ji) / 2`.

The warm residual is the saved Stage 8/9 evaluation residual. The final
residual is computed in memory as:

```text
L_frozen(final_vertices) - L_frozen(source_vertices)
```

using the saved directed graph weights. Source residual is zero by definition
because it is the reference state. No graph rebuild or solver call is needed.

## How to review quality

Visual checks are useful for finding problems, but a visually plausible page is
not proof of a good retarget. Review the same local frame across mesh, graph,
and residual modes, then compare numeric reports:

- lower interaction residual/objective, especially on hand vertices and contact
  frames, is generally better;
- collision penetration must remain zero/within the declared acceptance policy,
  with full-surface signed-distance audit passing;
- temporal continuity should have no spikes or jitter and should be compared
  against the source motion and the warm-start baseline;
- joint/base bounds, solver status, finite values, and artifact identity must
  pass the Stage 9/10 gates;
- compare algorithms on the same source window and robot using the same
  topology, sample identity, units, and reporting rules.

The viewer's residual max/mean/hand mean/object mean/top-k values are diagnostic
summaries. They should be used to localize errors, not as a replacement for the
formal Stage 9/10 acceptance reports.
