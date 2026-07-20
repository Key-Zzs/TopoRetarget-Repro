# Relative bone-direction initialization

Stage 7 implements the paper's Eq. (1) feature and exposes it as an initialization
trajectory. It does not implement object-relative positioning, interaction graphs,
Laplacian coordinates, collision queries, or final retargeting.

## Features

The default profile `mediapipe21_full_finger_chain_v1` constructs five directed
semantic chains:

```text
thumb  wrist -> thumb_cmc -> thumb_mcp -> thumb_ip -> thumb_tip
index  wrist -> index_mcp -> index_pip -> index_dip -> index_tip
middle wrist -> middle_mcp -> middle_pip -> middle_dip -> middle_tip
ring   wrist -> ring_mcp -> ring_pip -> ring_dip -> ring_tip
pinky  wrist -> pinky_mcp -> pinky_pip -> pinky_dip -> pinky_tip
```

They produce 20 directed bone vectors and the three consecutive pairs within
each finger produce 15 adjacent pairs. For a directed edge `e=(parent,child)`:

```math
v_e=p_{child}-p_{parent},\qquad d_e=v_e/\|v_e\|_2,
```

and the feature is `f_(e1,e2)=d_e1-d_e2`. The feature difference is not
renormalized, converted into an angle, or weighted by bone length. Eq. (1) is
the exact sum of squared residuals over all 15 pairs, with per-pair and
per-finger diagnostics retained.

`mediapipe21_phalange_only_diagnostic` excludes wrist-to-MCP edges and produces
15 bones/10 pairs. It is a bounded interpretation comparison, not an ablation
or a sequence-tuned choice.

## Wrist-centered frame

`canonical_keypoint_wrist_v1` is an explicit implementation assumption shared by
source and robot:

1. origin is semantic `wrist`;
2. `y` is the normalized wrist-to-middle-MCP direction;
3. `x` starts as index-MCP minus pinky-MCP, is Gram-Schmidt orthogonalized to
   `y`, and is normalized;
4. `z=x cross y`, normalized; `x=y cross z` is recomputed to close the
   orthonormality error;
5. the transform columns are `(x,y,z)`, so the determinant is `+1`.

The same formula is used for left and right hands. The side is retained in
provenance and semantic names are not mirrored or numerically re-indexed. The
result is mathematical right-handedness; the anatomical sign interpretation of
the third axis is audited rather than claimed to be paper-provided.

`translation_centered_scene_axes` subtracts the wrist but retains scene axes.
It is a diagnostic profile for testing whether the paper's wording only meant
translation centering. Strict degeneracy handling rejects short wrist-to-middle,
short index-to-pinky, and near-collinear axes with frame indices. There is no
identity-matrix or previous-frame fallback in strict mode.

## Invariance and limits

The default local profile is invariant to a common rigid translation and
rotation. It intentionally does not determine a floating base pose: local
directions contain no base translation information and the common base rotation
is removed with the derived frame. The solver therefore operates on the 22
Arti-MANO finger DoFs only. This is an implementation decision, not a claim
that the paper's displayed full `q` is wrong.

## CLI

```bash
toporetarget retarget inspect-bones \
  --canonical "$GRAB_CACHE" --hand right --frame 0 \
  --frame-profile canonical_keypoint_wrist_v1 \
  --bone-profile mediapipe21_full_finger_chain_v1 \
  --json .local/reports/stage7/source_bone_features_right.json \
  --csv .local/reports/stage7/source_bone_features_right.csv

toporetarget retarget compare-frame-profiles \
  --canonical "$GRAB_CACHE" --hand right --robot artimano_rh --frame 0 \
  --report .local/reports/stage7/frame_profile_comparison.json
```

See [WARM_START_OPTIMIZATION.md](WARM_START_OPTIMIZATION.md) for Eq. (2),
artifacts, validation, and solver behavior.
