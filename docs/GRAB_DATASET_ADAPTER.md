# GRAB dataset adapter

Stage 5 adds a production-oriented, read-only GRAB adapter. It is deliberately bounded: the
adapter indexes filenames lazily, loads one selected NPZ sequence, preserves the source time base
and native meshes, and writes an optional canonical Zarr cache. It does not perform full-batch
conversion or any later retargeting/geometry algorithm stage.

## Resource resolution

The adapter accepts an explicit `--grab-root` or sequence path. Otherwise it resolves `GRAB_ROOT`,
the ignored `.local/config.yaml` entry, the Stage 0 discovery report, and the registered local
storage layout. A valid root contains `grab/`, `tools/object_meshes/`, and sequence NPZ files.
The accepted local root used for the Stage 5 audit was the root resolved from the local discovery
report/configuration; it is intentionally not hard-coded in tracked files:

```text
${GRAB_ROOT}
```

The MANO root is separate, configured through `MANO_MODEL_ROOT`, and must contain
`MANO_LEFT.pkl` and `MANO_RIGHT.pkl`:

```text
$MANO_MODEL_ROOT
```

## Lazy index

`toporetarget data index --dataset grab` creates `.local/index/grab/index.jsonl` and
`manifest.json`. The index stores stable IDs (`s7/cubemedium_inspect_1`), source paths, subject,
object/action/repetition filename tokens, file size/mtime, and a root fingerprint. It does not
import MANO/SMPL-X, load frame arrays, or hash files unless `--hash-files` is supplied. Frame count,
native FPS, hand-field availability, mesh references, and contact availability are confirmed by
`describe` for the one selected NPZ.

Use `toporetarget data list` for bounded metadata queries and `data describe` for one sequence.
The index is a discovery/cache layer; the NPZ remains the source of truth for conversion.

## Canonical contract

`GrabDatasetAdapter.load_sequence` emits `HOISequence` data with:

- separate `right_hand` and `left_hand` tracks, each retaining native MANO16/SMPL-X joints,
  native 778-vertex scene geometry, and the derived wrist pose;
- optional `mediapipe21`, produced by the existing versioned Stage 3 semantic converter;
- object-local native mesh plus scene pose, and an optional `table` mesh as
  `support_surface` without treating the table as an interactable object;
- source contact arrays, binary `labels != 0` contacts, or an explicitly unavailable semantic
  mapping; and
- source timestamps, native FPS, contiguous half-open frame selection, and provenance hashes.

The adapter uses the personalized `vtemp` file referenced by each GRAB sequence when the source
provides one. It does not substitute a neutral MANO template. Object transforms follow the GRAB
row-vector convention while being stored in the canonical column-vector scene frame; the shared
transform is recorded in provenance and covered by raw/canonical comparison tests.

## Conversion and validation

```bash
toporetarget data index --dataset grab --output .local/index/grab
toporetarget data list --dataset grab --index .local/index/grab --subject s7 --limit 20
toporetarget data describe --dataset grab --index .local/index/grab \
  --sequence s7/cubemedium_inspect_1
toporetarget data convert --dataset grab --index .local/index/grab \
  --sequence s7/cubemedium_inspect_1 --hands both --start-frame 0 --end-frame 60 \
  --mano-model-root "$MANO_MODEL_ROOT" --contact-mode source \
  --output .local/cache/hoi/grab/s7/cubemedium_inspect_1/both_f000000_f000060.zarr
toporetarget data validate --dataset grab --index .local/index/grab \
  --sequence s7/cubemedium_inspect_1 --canonical \
  .local/cache/hoi/grab/s7/cubemedium_inspect_1/both_f000000_f000060.zarr \
  --mano-model-root "$MANO_MODEL_ROOT" --report .local/reports/stage5/grab_validation.json
```

Validation checks schema, source identity, frame range, timestamps, both hand sides, native
vertices/joints, optional MediaPipe21, wrist and bone round trips, object/table poses, contacts,
and optional raw/canonical comparison metrics. Missing semantic mappings and unavailable legacy
metrics are reported as unavailable rather than fabricated.

## Security and scope

GRAB NPZ files are treated as trusted local data. The reader uses `allow_pickle=True` only for this
explicit local dataset because the published GRAB files contain object-backed arrays; untrusted NPZ
files must not be supplied. The adapter never writes to the raw dataset root and creates caches
atomically at a caller-provided destination.

Object surface sampling, Delaunay/Laplacian interaction geometry, collision queries, SDFs,
MANO-to-Arti-MANO retargeting, training, and full-dataset conversion remain outside Stage 5.
