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
