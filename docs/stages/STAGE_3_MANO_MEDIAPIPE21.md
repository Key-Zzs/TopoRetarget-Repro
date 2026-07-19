# Stage 3 — explicit MANO to MediaPipe-style 21 conversion

Status: complete for the bounded source-hand adapter, with explicit MANO semantic and wrist-frame
assumptions retained. This stage accepts one canonical `HOISequence`/`HandTrack`, preserves every
source/object/timestamp field, and adds `keypoint_tracks["mediapipe21"]` in scene-frame metres.

## Delivered boundary

- YAML layout registry with validated `mediapipe21` topology and audited `mano16_smplx` source.
- Versioned MANO v1.2/SMPL-X mapping profile with named 16-joint mapping and five explicit mesh tips.
- Dense/sparse `J_regressor @ vertices` path, topology mismatch failures, validity propagation,
  overwrite protection, and no-detector/no-resampling provenance.
- Scene primary data plus derived scene↔wrist transforms; source wrist pose is not rewritten.
- CLI listing, profile description, cache conversion, validation, JSON/CSV reporting, static
  scene/wrist renders, and a local interactive sequence viewer.
- The interactive viewer provides a frame slider, previous/next buttons, scene/wrist switching,
  MANO mesh/source-joint/MediaPipe-21/skeleton/semantic-label/object-mesh/axes toggles, and
  current frame/timestamp/profile display. It only derives temporary display arrays.
- Synthetic tests and the real right-hand GRAB clip `cubemedium_inspect_1`, subject `s7`,
  `[0,60)`, 120 FPS, plus a real left-hand `[0,10)` smoke validation.

## Real acceptance

Input cache: `.local/cache/hoi/grab/cubemedium_inspect_1_rh_f000000_f000060.zarr`.

Output cache: `.local/cache/hoi/grab/cubemedium_inspect_1_rh_f000000_f000060_mp21.zarr`.

The output is `[60,21,3]`, scene-frame metres, with source `mano16` preserved. Direct joint and
tip consistency are zero; scene↔wrist round-trip is approximately `1.58e-16 m` RMSE and
`4.58e-16 m` max; timestamps, FPS, object mesh/pose, wrist pose, and source tracks are unchanged;
zero-length bone count is zero. The stored wrist-pose origin differs from MANO joint 0 by
`0.08606943 m`, documented under `A_HAND_FRAME_001`.

The left-hand output is `.local/cache/hoi/grab/cubemedium_inspect_1_lh_f000000_f000010_mp21.zarr`.
It contains `[10,21,3]`; direct copy and tip consistency are zero, scene↔wrist round-trip is
approximately `2.01e-16 m` RMSE and `4.71e-16 m` max, and source/object/timing/wrist integrity
checks pass. Its stored wrist-pose origin differs from MANO joint 0 by `0.08697721 m`; this is
also reported, not corrected, under `A_HAND_FRAME_001`.

Local evidence is under `.local/reports/stage3/`: backend/mapping audits, validation JSON/CSV,
source integrity, fingertip candidate comparison, right/left validation JSON/CSV, six first/middle/last
scene/wrist renders, and the interactive viewer smoke command.

## Explicit non-goals

Stage 3 does not implement Arti-MANO, robot FK/Jacobians, robot keypoint anchors, Eq. 1–9,
interaction graphs, object sampling, Delaunay, Laplacian/SDF/penetration constraints, RL/PPO,
full-dataset conversion, or another dataset adapter.
