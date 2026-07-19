# MANO to MediaPipe-style 21 conversion

Stage 3 is a source-hand adapter. It does not implement robot mapping, Arti-MANO, FK, interaction
graphs, optimization, Delaunay, SDF, or full-dataset conversion. The default profile is
`mano_v1_2_smplx_to_mediapipe21` version `1.0.0`.

## Explicit mapping

| Target index | Target semantic | Source type | Source semantic/index | Evidence | Assumption |
| ---: | --- | --- | --- | --- | --- |
| 0 | wrist | named MANO joint | `wrist` / 0 | MANO `J_regressor` and audited SMPL-X output | A_MANO_MEDIAPIPE_SEMANTICS_001 |
| 1 | thumb_cmc | named MANO joint | `thumb_cmc` / 13 | MANO kinematic order | A_MANO_MEDIAPIPE_SEMANTICS_001 |
| 2 | thumb_mcp | named MANO joint | `thumb_mcp` / 14 | MANO kinematic order | A_MANO_MEDIAPIPE_SEMANTICS_001 |
| 3 | thumb_ip | named MANO joint | `thumb_ip` / 15 | MANO kinematic order | A_MANO_MEDIAPIPE_SEMANTICS_001 |
| 4 | thumb_tip | mesh vertex | vertex 744 | installed `smplx.vertex_ids` MANO candidate | A_MANO_FINGERTIP_VERTICES_001 |
| 5 | index_mcp | named MANO joint | `index_mcp` / 1 | MANO kinematic order | A_MANO_MEDIAPIPE_SEMANTICS_001 |
| 6 | index_pip | named MANO joint | `index_pip` / 2 | MANO kinematic order | A_MANO_MEDIAPIPE_SEMANTICS_001 |
| 7 | index_dip | named MANO joint | `index_dip` / 3 | MANO kinematic order | A_MANO_MEDIAPIPE_SEMANTICS_001 |
| 8 | index_tip | mesh vertex | vertex 320 | installed `smplx.vertex_ids` MANO candidate | A_MANO_FINGERTIP_VERTICES_001 |
| 9 | middle_mcp | named MANO joint | `middle_mcp` / 4 | MANO kinematic order | A_MANO_MEDIAPIPE_SEMANTICS_001 |
| 10 | middle_pip | named MANO joint | `middle_pip` / 5 | MANO kinematic order | A_MANO_MEDIAPIPE_SEMANTICS_001 |
| 11 | middle_dip | named MANO joint | `middle_dip` / 6 | MANO kinematic order | A_MANO_MEDIAPIPE_SEMANTICS_001 |
| 12 | middle_tip | mesh vertex | vertex 443 | installed `smplx.vertex_ids` MANO candidate | A_MANO_FINGERTIP_VERTICES_001 |
| 13 | ring_mcp | named MANO joint | `ring_mcp` / 10 | MANO kinematic order | A_MANO_MEDIAPIPE_SEMANTICS_001 |
| 14 | ring_pip | named MANO joint | `ring_pip` / 11 | MANO kinematic order | A_MANO_MEDIAPIPE_SEMANTICS_001 |
| 15 | ring_dip | named MANO joint | `ring_dip` / 12 | MANO kinematic order | A_MANO_MEDIAPIPE_SEMANTICS_001 |
| 16 | ring_tip | mesh vertex | vertex 554 | installed `smplx.vertex_ids` MANO candidate | A_MANO_FINGERTIP_VERTICES_001 |
| 17 | pinky_mcp | named MANO joint | `pinky_mcp` / 7 | MANO kinematic order | A_MANO_MEDIAPIPE_SEMANTICS_001 |
| 18 | pinky_pip | named MANO joint | `pinky_pip` / 8 | MANO kinematic order | A_MANO_MEDIAPIPE_SEMANTICS_001 |
| 19 | pinky_dip | named MANO joint | `pinky_dip` / 9 | MANO kinematic order | A_MANO_MEDIAPIPE_SEMANTICS_001 |
| 20 | pinky_tip | mesh vertex | vertex 671 | installed `smplx.vertex_ids` MANO candidate | A_MANO_FINGERTIP_VERTICES_001 |

The thumb names are semantic approximations; they do not claim identical anatomical joint centers.
Non-tip points are copied by semantic name, not by an array-shape guess. Tip points are copied
exactly from the selected scene-frame mesh vertices. If vertices are missing or invalid, the tip
remains invalid rather than being fabricated.

The profile is tracked at `configs/keypoints/mappings/mano_v1_2_smplx_to_mediapipe21.yaml` and
implemented in `src/toporetarget/keypoints/mano_to_mediapipe.py`. It supports named joints plus
tip vertices, validated named MANO-21 reorder profiles, and vertices plus a dense or sparse local
MANO `J_regressor`. Model files remain external; their SHA-256 is recorded only in local reports
and output provenance.

## Frames and validation

`mediapipe21.positions_scene` is the sole canonical output. The wrist view is derived with the
existing SE(3) functions through `mediapipe21_scene_to_wrist` and
`mediapipe21_wrist_to_scene`; no wrist recentering, temporal resampling, interpolation, smoothing,
mirroring, or bone normalization is performed. Stage 2B's wrist-pose origin is preserved. In the
real `cubemedium_inspect_1` clip, MANO joint 0 is 0.08606943 m from that stored pose translation,
so `A_HAND_FRAME_001` remains unresolved and the value is reported rather than hidden.

Consistency metrics are implementation checks, not MediaPipe accuracy: direct joint/tip copy
RMSE/max, scene↔wrist round-trip, timestamp/frame/FPS preservation, object/wrist invariance,
source-track preservation, finite bones, zero-length bones, profile hash, and model hash.

Use `toporetarget keypoints layouts`, `profiles`, `describe-profile`, `convert`, `validate`, and
`visualize` for the explicit CLI boundary. Static PNG output is available with `--output`; local
interactive sequence inspection is available with `--show`, for example:

```bash
toporetarget keypoints visualize \
  --input <converted-zarr> --hand right --layout mediapipe21 --view scene \
  --start-frame 0 --end-frame 60 --show --show-source-layout --show-mesh --show-labels
```

The viewer has a frame slider, previous/next buttons, scene/wrist radio buttons, independent
MANO mesh, source MANO joints, MediaPipe-21, skeleton, labels, object mesh, and scene/wrist axes
toggles. Frame number, timestamp, and mapping profile ID are shown in the title. Viewer callbacks
never mutate canonical coordinates.
