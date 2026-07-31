# Stage 12 dataset adapters

Stage 12 validates four extracted datasets through one frozen contract:

```text
raw dataset -> CanonicalHOI v2 -> MediaPipe21 -> wuji_hand2_beta1_rh
                                      -> wuji_continuous_sequential_v1
```

The adapters are sequence-scoped and lazy. `discover`, `index`, and `describe`
read manifests or metadata only. `load_sequence` reads only the selected
frames, the selected MANO/object annotations, and the required mesh. No adapter
downloads, re-extracts, copies a complete dataset, resamples time, or applies
dataset-specific retargeting weights.

The Stage 12 integration worktree is `integration/dataset-adapter-v1`; the
corresponding local adapter branches are `feature/dataset-dexycb`,
`feature/dataset-oakink`, `feature/dataset-hocap`, and
`feature/dataset-contactpose`. Shared integration is limited to adapter
registration, the frozen selection runner, tests, and these docs; it does not
change robot, solver, viewer, or metric-core behavior.

| Dataset | Adapter | Raw contract | Selected trajectories |
| --- | --- | --- | --- |
| DexYCB | `DexYCBAdapterV1` | `labels_*.npz`, `meta.yml`, YCB mesh | pitcher, power drill |
| OakInk | `OakInkAdapterV1` | `seq_all.json`, `hand_v`, `obj_transf`, object mesh | two A01001 sequences |
| HO-Cap | `HOCapAdapterV1` | `poses_m.npy`, `poses_o.npy`, `meta.yaml`, object parts | G10, G04 |
| ContactPose | `ContactPoseAdapterV1` | annotation JSON, MANO fit JSON, object PLY | mug, banana |

The frozen selections and half-open frame ranges are in
[`configs/benchmarks/stage12_selection.yaml`](../configs/benchmarks/stage12_selection.yaml).
Every selected row produces a dataset manifest containing source path, version,
index hash, license status, sequence count, and capabilities.

## Canonical and provenance rules

All source MANO geometry is rendered with the shared MANO model and mapped to
MediaPipe21 through the existing semantic `mano_v1_2_smplx_to_mediapipe21`
converter. A shape-only joint reorder is not allowed. Source coordinates,
source hash, selected range, object IDs, and conversion convention are retained
in `CanonicalHOI v2` provenance.

ContactPose currently has no verified official hand-bone attribution in this
adapter. Its metadata therefore sets `contact_annotation_available=false` and
`contact_benchmark_status=NOT_AVAILABLE`; the adapter does not emit fabricated
Eq. 10/11 contact scores.

## Artifacts and reports

Outputs are isolated under
`.local/experiments/stage12_dataset_validation/<dataset>/<sequence>/`:

```text
canonical/canonical_hoi_v2.zarr
warm/warm_start.zarr
final/final_retarget.zarr
exports/object_samples.npz
exports/interaction_graph.zarr
exports/wuji_collision_samples.npz
metrics/retarget_report.json
metrics/retarget_report.md
html/source_warm_final_wuji.html
html/source_warm_final_wuji_smoke.json
```

Reports include warm/final Ebone, EIM/RMSE, per-finger RMSE, penetration,
continuity, runtime, solver/audit counts, and a failure category. HOCap keeps
all object parts in the canonical artifact. A multi-object selection must name
exactly one `primary_object`; the adapter assigns it
`primary_manipulation_object` and the shared Stage 8/9 path refuses to infer a
target from array order. The remaining declared parts are context geometry and
remain visible in source qualification HTML.

ContactPose is a one-frame static MANO fit in object coordinates. Its source
transform is `inv(mTc)`. A moving-hand `hTo` in the RGB-D annotations is kept
only as rigid-observation evidence and is never compounded into static MANO
vertices or joints.

Run the matrix with the repository environment:

```bash
PYTHONNOUSERSITE=1 \
  /home/deepcybo/miniconda3/envs/topo-retarget/bin/python \
  scripts/stage12_dataset_validation.py
```

Use `--dataset`, `--selection-index`, or `--max-trajectories` only for bounded
reruns. The script always writes under `.local`; it never commits generated
artifacts. After bounded runs complete, use `--aggregate-only` with no filters
to rebuild the eight-trajectory `stage12_summary.json` handoff (including Wuji
completion rate, report paths, HTML paths, and per-trajectory metrics) without
rerunning any solve. The opt-in adapter smoke test is enabled with
`STAGE12_RUN_NAS_TESTS=1` and is marked `licensed_data`.
