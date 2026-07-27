# Arti-MANO target-hand adapter

Stage 4 loads the two Arti-MANO URDFs as independent target-hand models. F0 distributes the
asset payload as a tracked vendor snapshot and does not copy ManipTrans Python code.

## Asset provenance and integrity

The tracked default is already available after checkout. To reproduce or update the vendor
snapshot from the pinned ManipTrans checkout:

```bash
toporetarget assets vendor-artimano \
  --source-root /home/deepcybo/workspace/dex/retarget/ManipTrans \
  --destination third_party/robot_hands/artimano \
  --imported-at 2026-07-27T19:00:00+08:00
toporetarget robots resolve-assets
```

The old `toporetarget assets import-artimano --destination .local/assets/artimano` command remains
available for local migration tests only. It is not the normal runtime path.

The tracked vendor snapshot used by the default registry has:

| Field | Value |
| --- | --- |
| upstream commit | `a3d08cfe3c3a5868a7f057533bcaf759c5af4705` |
| imported files | 98 |
| mesh files | 96 |
| tracked manifest SHA-256 | `c9601ed490bcec6f6d672d1ae4d8fd3f08724e357bf977cf63553a94cbdc3cf2` |
| source manifest SHA-256 | `1d14cce93e2ee09dedbfcda842b1d8aac29443f86b57a0a15f6289bd55e0f771` |
| tracked RH URDF SHA-256 | `422f8a229e8f22cf7989a5447cbe68014202e896c24e26407771f230596b671a` |
| tracked LH URDF SHA-256 | `9d83ed9cb3fd700a3d820582c3980b99e101ae44a768fb157c9af326c0e7bfbe` |
| source RH/LH URDF hashes | preserved in `SOURCE.yaml` |
| unresolved mesh references | 0 |
| asset modification | URDF path rebasing only |

The source ManipTrans checkout and legacy destination remain outside Git. F0 reports are in
`.local/reports/f0/`; the tracked bundle's manifest and provenance are under
`third_party/robot_hands/artimano/`.

## Configurations and topology

The tracked configs are `configs/robots/artimano_rh.yaml` and
`configs/robots/artimano_lh.yaml`; the shared profile is
`configs/robots/keypoints/artimano_mediapipe21.yaml`. Both sides have 28 links, 27 joints, 22
actuated joints, five fixed joints, root/base link `palm`, and fixed fingertip links
`thumb_tip`, `index_tip`, `middle_tip`, `ring_tip`, and `pinky_tip`.

The public qpos order is explicit and comes from the URDF plus ManipTrans's `artimano.py`; it is
not inferred from XML order:

| Finger | DoFs in public order |
| --- | --- |
| index | `j_index1y`, `j_index1z`, `j_index2`, `j_index3` |
| middle | `j_middle1y`, `j_middle1z`, `j_middle2`, `j_middle3` |
| pinky | `j_pinky1y`, `j_pinky1z`, `j_pinky2`, `j_pinky3` |
| ring | `j_ring1y`, `j_ring1z`, `j_ring2`, `j_ring3` |
| thumb | `j_thumb1x`, `j_thumb1y`, `j_thumb1z`, `j_thumb2y`, `j_thumb2z`, `j_thumb3` |

All 22 entries are revolute. The limits are identical in both URDFs:

| Joint family | Lower | Upper |
| --- | ---: | ---: |
| `*1y` (index/middle/pinky/ring) | -0.1745329252 | 0.1745329252 |
| `*1z` (index/middle/pinky/ring) | 0 | 1.5707963268 |
| `*2` | 0 | 1.7453292510 |
| `*3` | 0 | 1.3962634016 |
| `j_thumb1x` | 0 | 1.0471975512 |
| `j_thumb1y` | -0.2617993878 | 1.0471975512 |
| `j_thumb1z` | -1.0471975512 | 1.0471975512 |
| `j_thumb2y` | -0.1745329252 | 0.1745329252 |
| `j_thumb2z` | 0 | 1.5707963268 |
| `j_thumb3` | 0 | 1.3962634016 |

The neutral configuration is explicitly all-zero. The limits admit zero for every DoF; no task
specific ManipTrans pose is copied.

## MediaPipe-21-compatible anchors

The shared profile reuses Stage 3's `mediapipe21` names, indices, parent graph, and 20 edges:

| Index | Semantic | Anchor type | URDF source |
| ---: | --- | --- | --- |
| 0 | wrist | `link_origin` | `palm` |
| 1 | thumb_cmc | `joint_origin` | `j_thumb1x` |
| 2 | thumb_mcp | `joint_origin` | `j_thumb2y` |
| 3 | thumb_ip | `joint_origin` | `j_thumb3` |
| 4 | thumb_tip | `joint_origin` | `j_thumb_tip` |
| 5 | index_mcp | `joint_origin` | `j_index1y` |
| 6 | index_pip | `joint_origin` | `j_index2` |
| 7 | index_dip | `joint_origin` | `j_index3` |
| 8 | index_tip | `joint_origin` | `j_index_tip` |
| 9 | middle_mcp | `joint_origin` | `j_middle1y` |
| 10 | middle_pip | `joint_origin` | `j_middle2` |
| 11 | middle_dip | `joint_origin` | `j_middle3` |
| 12 | middle_tip | `joint_origin` | `j_middle_tip` |
| 13 | ring_mcp | `joint_origin` | `j_ring1y` |
| 14 | ring_pip | `joint_origin` | `j_ring2` |
| 15 | ring_dip | `joint_origin` | `j_ring3` |
| 16 | ring_tip | `joint_origin` | `j_ring_tip` |
| 17 | pinky_mcp | `joint_origin` | `j_pinky1y` |
| 18 | pinky_pip | `joint_origin` | `j_pinky2` |
| 19 | pinky_dip | `joint_origin` | `j_pinky3` |
| 20 | pinky_tip | `joint_origin` | `j_pinky_tip` |

The final mapping agrees with the audited candidate table. The `*1y`/`*1z` pairs and thumb
multi-axis groups are coincident URDF joint centers; the first named joint is used as the semantic
position. Fixed fingertip joint origins are used for tip anchors. RH and LH are loaded from their
own URDFs; LH's signed axes are not produced by mirroring RH in code. The stable profile hash is
`872900ba7c252562d0d84de7f75722d25b0026be238bc8e8af0cf088a909b04e`.

## Geometry coverage and commands

At neutral pose each side exposes 21 visual instances and 16 collision instances. The five fixed
tip links contain visual spheres but no collision geometry. Collision coverage is reported rather
than completed by replacing visual geometry; this is `A_ARTIMANO_COLLISION_COVERAGE_001`.

```bash
toporetarget robots list
toporetarget robots inspect --robot artimano_rh --json .local/reports/stage4/artimano_rh_inspect.json
toporetarget robots validate --robot artimano_rh --report .local/reports/stage4/artimano_rh_validation.json --csv .local/reports/stage4/artimano_rh_validation.csv
toporetarget robots fk --robot artimano_rh --pose neutral --dtype float64 --output .local/reports/stage4/artimano_rh_neutral_fk.json
toporetarget robots anchors --robot artimano_rh --csv .local/reports/stage4/artimano_rh_anchors.csv
toporetarget robots jacobian-check --robot artimano_rh --pose random --seed 4 --dtype float64 --report .local/reports/stage4/artimano_rh_jacobian.json
toporetarget robots visualize --robot artimano_rh --pose neutral --geometry visual --show-keypoints --show-skeleton --show-labels --show-base-frame --output .local/reports/stage4/artimano_rh_neutral_visual.png
toporetarget robots visualize --robot artimano_rh --pose neutral --geometry collision --show-keypoints --show-skeleton --output .local/reports/stage4/artimano_rh_neutral_collision.png
toporetarget robots visualize --robot artimano_rh --pose random --seed 4 --geometry both --show-keypoints --show-skeleton --show-labels --show-joint-axes --output .local/reports/stage4/artimano_rh_random_overlay.png
```

Use `artimano_lh` for the independent left-hand run. Stage 4 stops at `P^r(q)`, FK, anchor and
geometry inspection. It does not implement MANO-to-Arti-MANO qpos conversion, GRAB ingestion,
bone initialization, Delaunay/Laplacian/SDF, collision optimization, or RL.

## Stage 7.1 thumb audit

The read-only warm-start audit consumes the accepted RH manifest and verifies
that the `mediapipe21` semantic thumb chain resolves to the declared
`j_thumb1x` → `j_thumb2y` → `j_thumb3` → `j_thumb_tip` ancestry. Joint axes and
limits are evidence for diagnosis only. A morphology-normalized target is never
used to rewrite this adapter or the formal Stage 7 source target.
