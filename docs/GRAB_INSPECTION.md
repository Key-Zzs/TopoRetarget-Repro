# Historical Stage 2B GRAB inspection

This document records the historical one-sequence Stage 2B inspection. The production-oriented
Stage 5 adapter, lazy index, bimanual conversion, validation, contacts, and interactive viewer are
documented in [`GRAB_DATASET_ADAPTER.md`](GRAB_DATASET_ADAPTER.md) and
[`GRAB_INTERACTIVE_VISUALIZATION.md`](GRAB_INTERACTIVE_VISUALIZATION.md).

Stage 2B implements a bounded inspection adapter, not a full GRAB dataset adapter. Every command
requires one explicit `.npz` sequence path; there is no subject/sequence enumeration, batch index,
full-dataset conversion, source rewrite, extraction, contact clipping, FPS conversion, or spatial
sampling.

## Resources and fields

Resource precedence is CLI (`--grab-root`, `--mano-model-root`), environment (`GRAB_ROOT`,
`MANO_MODEL_ROOT`), `.local/config.yaml`, then GRAB-root inference from the selected sequence's
ancestors. Tracked examples contain placeholders only.

The reader preserves gender, subject ID, object name, motion intent, native framerate, frame count,
selected hand parameters and personalized vtemp, object parameters and mesh, table metadata, and
contact metadata. Contact arrays are registered but not loaded or attributed during inspection.
If source vertices are present, they are used directly; otherwise the replaceable SMPL-X/MANO
backend reconstructs the selected hand from full `fullpose` (45 axis-angle values) when available.

GRAB's official object helper applies row-vector `v @ R`. The adapter records the native GRAB scene
as `S` and stores the equivalent column-vector object pose with `R.T`. Hand and object are therefore
in the same scene frame. Positions/meshes are metres, axis-angle is radians, and time is seconds.

## Frame range and commands

`--start-frame` is inclusive and `--end-frame` is exclusive. This selects a contiguous clip; it
does not interpolate or resample. `native_fps` is retained as metadata. `--display-stride` changes
only which frames are displayed.

```bash
export GRAB_SEQUENCE=/path/to/one/grab/subject/sequence.npz
export GRAB_ROOT=/path/to/the/GRAB/root
export MANO_MODEL_ROOT=/path/to/MANO/models

toporetarget data describe --dataset grab --sequence-path "$GRAB_SEQUENCE" \
  --grab-root "$GRAB_ROOT" --hand right --mano-model-root "$MANO_MODEL_ROOT"

toporetarget data convert --dataset grab --sequence-path "$GRAB_SEQUENCE" \
  --grab-root "$GRAB_ROOT" --hand right --mano-model-root "$MANO_MODEL_ROOT" \
  --start-frame 0 --end-frame 60 \
  --output .local/cache/hoi/grab/sequence_rh_f000000_f000060.zarr

toporetarget data compare --dataset grab --sequence-path "$GRAB_SEQUENCE" \
  --grab-root "$GRAB_ROOT" --hand right --mano-model-root "$MANO_MODEL_ROOT" \
  --canonical .local/cache/hoi/grab/sequence_rh_f000000_f000060.zarr \
  --layout side-by-side --frame 0 \
  --output .local/reports/stage2b/grab_side_by_side.png \
  --error-json .local/reports/stage2b/grab_side_by_side.json \
  --error-csv .local/reports/stage2b/grab_side_by_side.csv

toporetarget data compare --dataset grab --sequence-path "$GRAB_SEQUENCE" \
  --grab-root "$GRAB_ROOT" --hand right --mano-model-root "$MANO_MODEL_ROOT" \
  --canonical .local/cache/hoi/grab/sequence_rh_f000000_f000060.zarr \
  --layout overlay --frame 0 \
  --output .local/reports/stage2b/grab_overlay.png \
  --error-json .local/reports/stage2b/grab_overlay.json \
  --error-csv .local/reports/stage2b/grab_overlay.csv

toporetarget data compare --dataset grab --sequence-path "$GRAB_SEQUENCE" \
  --grab-root "$GRAB_ROOT" --hand right --mano-model-root "$MANO_MODEL_ROOT" \
  --canonical .local/cache/hoi/grab/sequence_rh_f000000_f000060.zarr \
  --layout overlay --start-frame 0 --end-frame 60 --display-stride 1 --show
```

The raw side is reconstructed directly from the NPZ/backend and the canonical side is loaded from
the separate Zarr cache. JSON and CSV contain per-frame values plus mean, median, p95, and max
summaries. Unavailable metrics are explicitly marked unavailable rather than filled with zero.

## Current local result and limitations

The bounded local inspection selected the anonymous sequence ID `cubemedium_inspect_1`, right hand,
clip `[0, 60)`, native FPS `120.0`. `describe` passed and object/personalized-vtemp paths resolved.
The user-provided MANO root contains the official `MANO_LEFT.pkl` and `MANO_RIGHT.pkl` files; the
optional SMPL-X package loads those MANO models, so this acceptance uses real MANO reconstruction,
not a spatial crop from an SMPL-X body mesh.

The real conversion and raw-to-canonical comparison passed. The canonical cache is
`.local/cache/hoi/grab/cubemedium_inspect_1_rh_f000000_f000060.zarr`; the required reports are
`.local/reports/stage2b/grab_side_by_side.png`, `grab_overlay.png`, and their JSON/CSV metrics.
Additional side-by-side renders were produced for frames 30 and 59. Across the 60-frame clip,
hand vertex/keypoint, wrist translation, object translation/world-vertex, and timestamp errors are
zero; maximum rotation error is approximately `1.71e-6` degrees, from floating-point SE(3)
comparison. Source hash and mtime are unchanged. This remains a one-sequence, 60-frame inspection,
not a full GRAB conversion or a claim that later retargeting stages are complete.

Stage 3 consumes this Stage 2B cache as an immutable source and writes a separate cache with an
explicit `mediapipe21` track. It does not overwrite the inspection cache or the GRAB NPZ. See
[`MANO_TO_MEDIAPIPE21.md`](MANO_TO_MEDIAPIPE21.md) for the profile and
`toporetarget keypoints convert` for the bounded conversion command.
