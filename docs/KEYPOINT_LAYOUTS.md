# Keypoint layouts

Stage 3 defines semantic hand layouts independently of detector software. `mediapipe21` means
the MediaPipe Hands-compatible 21-point names, indices, and tree only. These points are produced
from MANO geometry (`coordinate_source: mano_geometry`); they are not MediaPipe detector
predictions, MediaPipe world landmarks, or a MediaPipe accuracy measurement.

## `mediapipe21`

The canonical output is scene-frame metres with shape `[T, 21, 3]`:

| Index | Semantic | Parent |
| ---: | --- | --- |
| 0 | wrist | — |
| 1 | thumb_cmc | 0 |
| 2 | thumb_mcp | 1 |
| 3 | thumb_ip | 2 |
| 4 | thumb_tip | 3 |
| 5 | index_mcp | 0 |
| 6 | index_pip | 5 |
| 7 | index_dip | 6 |
| 8 | index_tip | 7 |
| 9 | middle_mcp | 0 |
| 10 | middle_pip | 9 |
| 11 | middle_dip | 10 |
| 12 | middle_tip | 11 |
| 13 | ring_mcp | 0 |
| 14 | ring_pip | 13 |
| 15 | ring_dip | 14 |
| 16 | ring_tip | 15 |
| 17 | pinky_mcp | 0 |
| 18 | pinky_pip | 17 |
| 19 | pinky_dip | 18 |
| 20 | pinky_tip | 19 |

The 20 edges are `(0,1),(1,2),(2,3),(3,4)`, `(0,5),(5,6),(6,7),(7,8)`,
`(0,9),(9,10),(10,11),(11,12)`, `(0,13),(13,14),(14,15),(15,16)`, and
`(0,17),(17,18),(18,19),(19,20)`. Fingertips are `[4, 8, 12, 16, 20]`.

Both left and right hands use this same semantic order. A left hand is not mirrored, and hand side
does not change mathematical frame handedness.

## Registered source layout

`mano16_smplx` is the explicit source layout emitted by the current SMPL-X/MANO backend:
`wrist`, index joints 1–3, middle joints 1–3, pinky joints 1–3, ring joints 1–3, and thumb
joints 1–3. The legacy Stage 2B cache label `mano16` is an explicit alias for this audited
source layout; it is not renamed to `mediapipe21`.

Stage 7 reuses this semantic order without scattering integer indices. Its full
finger-chain profile produces 20 directed edges and 15 consecutive within-finger
pairs; the phalange-only profile produces 15 edges and 10 pairs for a bounded
diagnostic comparison. Both are configured under `configs/retarget/bones/`.

## Configuration and extension

Tracked definitions are in `configs/keypoints/layouts/mediapipe21.yaml` and
`configs/keypoints/layouts/mano16_smplx.yaml`. They are loaded through
`src/toporetarget/keypoints/registry.py` and validated for unique names, contiguous indices,
one parent per non-root point, an acyclic graph, and edge/parent agreement. Add a new YAML layout,
give it a distinct name/version, document its source ordering and evidence, then add tests before
using it in a mapping profile.

The Stage 5 GRAB adapter reuses this registered Stage 3 converter when `--include-mediapipe21` is
requested. The native `mano16_smplx` track remains present and is never silently reshaped into
the 21-point semantic layout.
