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
- source, binary, and explicitly unavailable semantic contact modes;
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

Stage 6 is not started. No object 50-point sampling, Delaunay/Laplacian graph, SDF/collision
query, MANO-to-Arti-MANO retargeting, PPO, or full-batch conversion is included.
