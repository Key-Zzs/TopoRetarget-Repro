# Multi-Dataset Interaction Benchmark

This document defines the bounded Q1 selection contract. The versioned schema is
`toporetarget.hoi_benchmark.v1`; its local configuration is
`configs/benchmarks/hoi_benchmark_v1.yaml`.

## Units

GRAB units are 60 contiguous native frames with native timestamps and no resampling. The retained
legacy unit is `s1/airplane_lift`, right hand, global `[240,300)`. Additional units are selected
from lazy filename metadata, native contact labels, personalized MANO/vtemp availability, bounded
source checks, and a strict reference-winding object mesh audit.

ContactPose units are native single grasps. They are represented as `T=1` only when the canonical
adapter requires a time dimension; no pose is copied to manufacture a trajectory. Temporal metrics
are `NOT_APPLICABLE`, not zero. `mug`, `scissors`, and Utah teapot are a diagnostic exclusion set,
never deleted or edited.

## Freeze and integrity

`benchmark_selection_manifest.json` records IDs, source/object/contact hashes, frame ranges, scores,
all candidate rejection reasons, the Git commit, and a manifest hash. `benchmark_selection.lock`
binds all runs to that hash. A failed unit is preserved as a failure; result-based replacement or
frame-range edits are forbidden. A data-identity correction requires a new manifest version.

The current local selection audit is intentionally not frozen: the GRAB index contains 1,334
non-fixed candidates, of which 16 were evaluated under the bounded NAS probe and all 16 passed the
declared contact gates; ContactPose produced 110 candidate annotation records and selected 0.
All 110 lacked recognized official contact attribution (12 also belong to the declared diagnostic
exclusion set). The generated status is `Q1_CONTACTPOSE_SELECTION_BLOCKED`, so baseline execution
and result-level evaluation remain blocked until an official attribution-bearing ContactPose
snapshot is supplied.

## Diversity and aggregation

The selector prefers thumb/index precision, multi-finger support, non-tip/palm coverage, distinct
objects/subjects, and a left-hand candidate when it does not violate hard conditions. Per-dataset
macro metrics weight units equally. Dynamic and static units are reported separately, exact
ContactPose metrics are not mixed with GRAB proxies, and no single black-box score is constructed.
