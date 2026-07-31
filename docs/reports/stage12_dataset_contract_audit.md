# Stage12 dataset contract forensic audit

Date: 2026-07-31  
Scope: raw dataset → adapter → Canonical HOI v2 → MANO reconstruction → source visualization → retarget input.  
Method boundary: diagnosis only. No solver, Wuji adapter, Stage12 v4 profile, optimization weight, final-refinement, or metric code was run or changed.

## Verdict

The first corruption is in dataset-adapter MANO reconstruction, before Canonical HOI wrapping:

- DexYCB and HO-Cap store 51 values as global axis-angle (3), PCA-45 coefficients, and translation (3), with subject betas. Stage12 labels the middle 45 values `fullpose`, selects non-PCA MANO, forces `flat_hand_mean=True`, and omits subject betas.
- ContactPose stores one PCA-15 MANO fit and betas per hand/grasp. Its frame composition is correct, but the same backend forces the wrong hand mean and omits betas.
- OakInk works because the adapter passes author-provided `hand_v` vertices through without reconstructing MANO; `obj_transf` is also copied directly.
- The canonicalization method only validates and wraps the already-created arrays. It does not introduce or correct the error.

All eight selected object-pose arrays match their raw annotations exactly (`max_abs=0`). No evidence supports a missing inverse, camera/world confusion, left/right mirror, axis swap, or unit-scale error as the primary cause.

## Contract table

| Dataset | Native hand / MANO | Native object | Scene and units | Temporal contract | Required transforms | First bad stage |
|---|---|---|---|---|---|---|
| DexYCB | `pose_m[51]`: global AA3 + PCA45 + translation3; subject `betas[10]`; raw `joint_3d[21,3]` | `pose_y[3,4] = T_C_O` | Selected RGB-D camera `C`, metres | Synchronized trajectory | `S := C`; no extra rigid transform | Adapter MANO decode |
| HO-Cap | `poses_m[H,T,51]`: global AA3 + PCA45 + translation3; subject `betas[10]` | `poses_o[O,T,7]`: qxyzw + translation, `T_W_O` | Capture world `W`, metres | Multi-hand/multi-object trajectory | `S := W`; quaternion conversion only | Adapter MANO decode |
| ContactPose | One `pose[18]`: global AA3 + PCA15, `betas[10]`, `mTc` per valid hand/grasp | Object PLY is canonical `O`; frame JSON has `hTo`, `oTw` | Object coordinates `O`, metres | Static articulated grasp; optional rigid hand motion and RGB-D frames | `hTm=inv(mTc)`; moving: `oTh=inv(hTo)`; `oTm=oTh@hTm` | Adapter MANO decode; Stage12 60-frame selection also violates static-unit contract |
| OakInk | Per-frame author `hand_v[778,3]` and `hand_j[21,3]` | `obj_transf[4,4] = T_C_O` | Camera `C`, metres | Image-sequence trajectory | `S := C`; transform object mesh by `T_C_O` | None in source mesh/pose path |

The machine-readable form is [dataset_contract_table.csv](../../.local/reports/stage12_adapter_forensic_audit/dataset_contract_table.csv).

## DexYCB

Native contract:

- Per-camera `labels_*.npz` contains `pose_m`, `joint_3d`, and `pose_y`.
- `pose_m[:,0:48]` is MANO pose in PCA representation and `pose_m[:,48:51]` is translation; the sequence `meta.yml` points to subject MANO betas.
- `pose_y` is the model-to-camera `[R|t]` transform. The Stage12 object path preserves it exactly.
- Stage12 selected only right-hand sequences. The raw 21-joint order is wrist, thumb, index, middle, ring, little.

Trace:

```text
hand:   pose_m in selected camera C
        -- MANO(global AA3, PCA45, betas10, translation3) --> vertices in C
        -- T_S_C = I --> canonical S

object: YCB local O -- pose_y = T_C_O --> C -- T_S_C = I --> canonical S
```

Observed first corruption:

```text
render_mano_fullpose(pose_m)
  middle 45 values passed as full axis-angle
  -> SmplxManoBackend chooses use_pca=False
  -> flat_hand_mean=True and no betas
```

Across the two selected 60-frame clips, current-versus-native MANO vertex error is 28.15/33.95 mm mean, 68.75/78.00 mm p95, and 90.73/94.53 mm maximum. The object transform remains exact. This matches the official [DexYCB toolkit contract](https://github.com/NVlabs/dex-ycb-toolkit) and its [dataset loader](https://github.com/NVlabs/dex-ycb-toolkit/blob/master/dex_ycb_toolkit/dex_ycb.py).

## HO-Cap

Native contract:

- `poses_m.npy` is `[hand,time,51]`; the hand vector has the same AA3 + PCA45 + translation3 convention.
- Per-subject `data/calibration/mano/subject_N.yaml` supplies `betas[10]`.
- `poses_o.npy` is `[object,time,7]`; Stage12 correctly transposes to `[time,object,7]`, interprets qxyzw through SciPy, and keeps translation in metres.
- The selected sequences are genuine dynamic trajectories.

Trace:

```text
hand:   poses_m in world W
        -- MANO(global AA3, PCA45, subject betas10, translation3) --> W
        -- T_S_W = I --> canonical S

object: local O -- qxyzw + t = T_W_O --> W -- T_S_W = I --> canonical S
```

The two clips show 29.36/29.01 mm mean, 74.20/73.61 mm p95, and 86.67/86.82 mm maximum vertex error. Every raw object matrix again matches the adapter matrix exactly. The native reconstruction contract is also explicit in HO-Cap's official [`ManoLayer`](https://github.com/IRVLUTD/HO-Cap/blob/main/hocap_toolkit/layers/mano_layer.py) and [`MANOGroupLayer`](https://github.com/IRVLUTD/HO-Cap/blob/main/hocap_toolkit/layers/mano_group_layer.py): PCA45, non-flat mean, subject betas, then translation.

## ContactPose

Native contract:

- A ContactPose record is one grasp/contact configuration. `mano_fits_15.json` contains one PCA-15 MANO fit and one beta vector per valid hand, not per-frame articulation.
- `annotations.json` may contain many RGB-D frames. A `moving=true` hand has per-frame rigid `hTo`; it does not acquire a new per-frame MANO pose.
- The object PLY carries contact intensity in vertex color. The mug and banana files have 220 and 183 distinct intensity values respectively.
- Official composition is exactly `hTm=inv(mTc)`, `oTh=inv(hTo)` when moving, then `oTm=oTh@hTm`. Stage12 implements these inverse directions correctly.

Trace:

```text
MANO M -- hTm=inv(mTc) --> hand H
       -- oTh=inv(hTo), or I for static hand --> object O
       -- S := O --> canonical S

object mesh O -- I --> canonical S
```

The fixed mug is exactly static over the selected 60 frames. The banana has only rigid object-relative motion; the same MANO articulation is reused. Current-versus-native-contract vertex error is 24.31 mm mean for mug and 27.38 mm for banana because `flat_hand_mean=True` and missing betas alter the one source grasp before it is transformed.

Classification: **ContactPose is static contact evaluation only.** Do not force a 60-frame trajectory. Use `T=1`; mark temporal metrics `NOT_APPLICABLE`. This agrees with the repository's existing benchmark contract in `docs/MULTI_DATASET_INTERACTION_BENCHMARK.md` and the dataset's description as unique grasps in the official [ContactPose repository](https://github.com/facebookresearch/ContactPose) and [project site](https://contactpose.cc.gatech.edu/).

## OakInk

Native contract and trace:

```text
hand:   per-frame hand_v already in camera C -- T_S_C=I --> canonical S
object: local O -- obj_transf=T_C_O --> C -- T_S_C=I --> canonical S
```

Both selected clips have exactly 0 mm adapter-versus-source vertex error and raw-versus-adapter object-transform error of zero. This explains why the current HTML looks plausible: there is no MANO parameter reinterpretation. OakInk-Image provides dynamic annotated clips, consistent with the official [OakInk repository](https://github.com/oakink/OakInk).

The adapter does set an identity `wrist_pose_scene` despite storing scene-space vertices and discards raw `hand_j`; these are secondary contract issues for downstream keypoints/pose metadata, not the source mesh/object failure.

## Coordinate failure checks

| Check | DexYCB | HO-Cap | ContactPose | OakInk |
|---|---:|---:|---:|---:|
| Missing inverse | No | No | No; both required inverses verified | No inverse required |
| Camera/world confusion | No | No | No; canonical is object frame | No |
| Left/right mirror | No | No | No | No |
| Axis swap | No | No | No | No |
| Unit conversion | metre→metre | metre→metre | metre→metre | metre→metre |

Full first/last matrices, determinants, and per-selection raw equality checks are in [coordinate_trace.json](../../.local/reports/stage12_adapter_forensic_audit/coordinate_trace.json).

## MANO and keypoint semantics

The configured semantic order is structurally correct:

- MANO16: wrist, index, middle, pinky, ring, thumb (three joints per finger).
- MediaPipe21: wrist, thumb, index, middle, ring, pinky (four points per finger).
- Wrist remains index 0; target thumb indices are 1–4; fingertip target indices are 4, 8, 12, 16, 20.
- The profile uses semantic names, not a positional reshape, and does not mirror either hand.

However, `_mano_native_track` reapplies the rest MANO `J_regressor` to already posed vertices. It ignores the posed joints returned by the MANO backend and ignores dataset-native `joint_3d`/`hand_j`. Thus semantic ordering passes but native joint preservation fails. ContactPose also uses its own official fingertip vertices `[333,444,672,555,745]` in MANO internal finger order before OpenPose reordering, whereas the repository profile uses SMPL-X anchors `[320,443,671,554,744]` for index/middle/pinky/ring/thumb.

The complete audit and validation alias are [mano_mapping_audit.json](../../.local/reports/stage12_adapter_forensic_audit/mano_mapping_audit.json) and [mapping_validation.json](../../.local/reports/stage12_adapter_forensic_audit/mapping_validation.json).

## Source-only evidence

[source_only_html/index.html](../../.local/reports/stage12_adapter_forensic_audit/source_only_html/index.html) links all eight selected clips. Each page contains only:

- native-contract MANO (or native OakInk `hand_v`),
- current adapter MANO for comparison,
- object mesh and per-frame object pose,
- native/raw joints when available,
- ContactPose contact-intensity points when available.

No Wuji, warm-start, final-retarget, or solver result is embedded.

## Fix Implementation

Stage 12.5 implements the remediation identified above without changing the
Wuji/Arti-MANO plugins, SDF/compiled-sign backends, solver objectives, or any
raw dataset file.  The repair is source-only: the corrected adapters and their
new artifacts are under `.local/experiments/stage12_source_contract_fix_v1/`.
The Stage12 final queue remains operator-paused and no warm/final retarget was
run as part of this remediation.

## Explicit MANO Contract

`toporetarget.mano.reconstruction.v2` replaces representation inference with
an explicit `ManoReconstructionRequest` and `ManoReconstructionResult`.
Requests declare side, representation, PCA component count, float64/metre
units, `flat_hand_mean`, beta requirements, model/source hashes, and the
complete source provenance.  The backend rejects ambiguous legacy calls,
invalid dimensions, absent calibrated betas, unsupported sides, and incomplete
provenance.  PCA15 and PCA45 are expanded with the exact layer basis/mean;
PCA45 is never interpreted as a 45D axis-angle hand pose.  The result retains
the derived 48D axis-angle form, posed joints, faces, beta broadcast, and the
model/basis/mean hashes.  Layer identity includes side, representation, K,
flat-hand-mean, model hash, dtype, and device.

## Dataset Adapter Changes

- DexYCB now declares `pose_m = AA3 + PCA45 + translation3`, uses the
  sequence `mano_calib` subject beta vector, preserves raw `joint_3d`, and
  semantically reorders it to MediaPipe21.  Its `pose_y` object matrices are
  direct passthrough.
- HO-Cap now declares the same PCA45 contract, loads the sequence subject's
  MANO calibration beta vector, retains the backend's posed MANO16 joints,
  and preserves the existing `[object,time,7] -> [time,object,7]` qxyzw path.
- ContactPose reconstructs one PCA15 grasp with fit betas and
  `flat_hand_mean=false`, then applies `hTm=inv(mTc)` and, only for a moving
  observation, `oTh=inv(hTo); oTm=oTh@hTm`.
- OakInk remains a direct `hand_v` and `obj_transf` passthrough, but now
  preserves raw `hand_j` as the native joint source.  Its wrist translation is
  sourced from raw joint zero and orientation is explicitly unavailable,
  causing orientation-dependent use to fail closed.

## Native Joint Preservation

No Stage12 adapter silently applies a rest MANO `J_regressor` to posed
vertices as its canonical source joint track.  DexYCB and OakInk use their
dataset-native 21-point arrays, HO-Cap uses backend forward posed MANO16
joints, and ContactPose uses its documented MANO16 plus official fingertip
vertex convention.  The eight-selection native and canonical joint checks are
exact (`max=0 m`); their manifests are in
`.local/reports/stage12_source_contract_fix/native_joint_report.json`.

## ContactPose Static Contract

Both frozen ContactPose selections are now
`static_contact_evaluation_only`, with `frame_count=1`,
`articulated_frame_count=1`, `temporal_metrics=NOT_APPLICABLE`, and no repeated
pose manufacturing.  The banana selection retains only an explicitly labelled
rigid-observation diagnostic; that observation is not an articulated source
trajectory.  Per-selection static evidence is stored alongside each corrected
source report.

## Eight-Selection Requalification

All eight frozen choices were requalified against an independent direct
SMPL-X/MANO reference (not the adapter backend).  Adapter/reference vertices,
canonical joints, native joints, and raw object transforms each have zero
measured maximum error in the current bounded real-data run.  Dynamic viewers
were loaded and captured at frames 0/15/30/45/59; static ContactPose viewers at
frame 0.  Browser error count is zero and the image validator confirms a
nonempty canvas plus both hand and object layers.  The authoritative summary is
`.local/reports/stage12_source_contract_fix/source_qualification_summary.json`.

## Remaining Limitations

This repair proves the source-adapter contracts only.  It does not certify a
new Wuji/solver trajectory, collision outcome, warm start, final refinement,
or downstream retarget acceptance.  Hand-object proximity values in the new
reports remain labelled `ENGINEERING_DIAGNOSTIC`; they are not contact ground
truth.

## Stage12 v4 Invalidation

Historical Stage12 v4 artifacts remain preserved but are not formally usable:
DexYCB and HO-Cap source/canonical/warm/final/metrics/HTML depend on the wrong
MANO interpretation; ContactPose's manufactured 60-frame trajectory is
invalid; and OakInk's mesh remains valid but its native-joint-dependent
retarget input requires regeneration.  The machine-readable statement is
`.local/reports/stage12_source_contract_fix/stage12_v4_invalidation.json`.

## Required Downstream Regeneration

After an explicit user approval, regenerate the affected Stage12 source and
then downstream canonical/warm/final outputs from the new qualified source
artifacts.  Do not overwrite the preserved v4 evidence, and do not resume the
paused final queue until that separate authorization is given.
