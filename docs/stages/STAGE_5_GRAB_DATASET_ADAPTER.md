# Stage 5 — production GRAB dataset adapter

## Objective

Implement a production-oriented GRAB dataset adapter with a lazy index, explicit single-sequence
conversion, native-time/native-mesh preservation, validation, provenance, raw/canonical comparison,
and an interactive HOI viewer.

## Delivered

- filename-first index and metadata query CLI;
- explicit right, left, or bimanual sequence loading;
- personalized MANO `vtemp` resolution and existing SMPL-X/MANO backend reuse;
- native MANO16/SMPL-X joints, 778-vertex hand geometry, object/table meshes and poses;
- source, binary, and official semantic contact modes;
- optional Stage 3 MediaPipe21 tracks;
- atomic Zarr cache creation and source integrity metadata;
- validation JSON/CSV reports and raw/canonical error metrics;
- raw/canonical/compare static rendering and bounded interactive controls; and
- English/Chinese documentation of assumptions and non-goals.

## Acceptance boundary

The bounded real acceptance sequence is `s7/cubemedium_inspect_1`, using the locally configured
GRAB and MANO roots described in `docs/GRAB_DATASET_ADAPTER.md`. Right-hand and both-hand `[0, 60)` clips
convert at the source 120 Hz. Validation reports zero timestamp/contact/vertex translation error,
floating-point-level wrist/object rotation error, and preserved source hashes. The legacy Stage 2B
cache comparison has no native-keypoint metric because that cache does not contain the formal
native-keypoint field; this is reported as unavailable, not as a pass by substitution.

The official semantic contact mapping is verified from `otaheri/GRAB/tools/utils.py` at commit
`4dab3211fae4fc5b8eb6ab86246ccc3a42d8f611` and is tracked with its source SHA-256 in
`configs/datasets/grab_contact_parts.yaml`. Strict conversion accepts labels `0..55`; non-strict
conversion maps out-of-range labels to explicit semantic ID `56` and records them. A fresh MANO-backed
semantic rerun remains dependent on the local MANO model files being configured.

Stage 6 is not started. No object 50-point sampling, Delaunay/Laplacian graph, SDF/collision
query, MANO-to-Arti-MANO retargeting, PPO, or full-batch conversion is included.

## Closeout evidence and final boundary

The adapter reuses the canonical Stage 2 HOI schema/storage and the existing Stage 2/5 viewer
artist/update infrastructure; semantic contacts add a representation to that viewer rather than
introducing a parallel renderer. Conversion remains lazy and sequence-scoped: the filename-first
index stores metadata only, and a caller may create one optional per-sequence Zarr cache. Whole-dataset
materialization is intentionally not performed because it would load external MANO assets and large
frame/mesh arrays without being required by the adapter contract.

The real evidence uses at most two 60-frame clips:

- `s7/cubemedium_inspect_1`, `[0,60)`, 120 Hz, right and bimanual existing MANO-backed geometry,
  object `cubemedium`, table, and separate left/right tracks. This clip is contact-free in the
  selected range and is retained as the geometry/right/bimanual regression.
- `s1/airplane_lift`, `[238,298)`, 120 Hz, right hand, object `airplane`, table, and observed labels
  `[0,43,46,55]`. Source, binary, and semantic contact caches all preserve the same raw labels;
  semantic validation reports no unmapped labels and exact raw/binary/semantic/mapping round trips.

The semantic contact viewer has headless first/middle/last PNGs, binary and both-hand/table PNGs,
cache-enrichment compare layouts, a semantic legend, mapping ID/version in the title, and an
explicit `interactive_viewer_smoke.json` where slider, first/last, previous/next, play/pause,
keyboard, four reference frames, source/visibility/contact controls, 100 updates, and timer close
are individually marked. `source_integrity.json` records NPZ/object/table/vtemp hashes and current
MANO-file availability.

The official mapping, schema, viewer, CLI alias, index, docs, default tests, and fresh real-data
closeout are complete. Fresh MANO-backed conversions for both clips were run with an explicit
external MANO root, and validation passed with exact raw/binary/semantic/mapping round trips,
fully mapped official labels, native timestamps, and unchanged source/object/table/model hashes.
The external root remains a runtime input rather than a tracked repository asset.
