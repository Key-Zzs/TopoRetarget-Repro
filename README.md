# TopoRetarget-Repro

[中文 README](README.zh-CN.md)

TopoRetarget-Repro is an unofficial, independent, paper-traceable reproduction repository for
[*TopoRetarget: Interaction-Preserving Retargeting for Dexterous Manipulation*](https://arxiv.org/abs/2606.16272).
It provides a robot-independent HOI data contract, explicit coordinate conventions, source-hand
conversion tools, reproducibility audits, and a staged path toward full dexterous retargeting.

The repository is intentionally transparent about scope: the current implementation reaches the
canonical HOI interface, a bounded MANO-to-MediaPipe-style-21 source adapter, the Stage 4 generic
robot-hand/Arti-MANO target kinematics interface, and the bounded Stage 5 GRAB dataset adapter. It
does not yet claim to implement the paper's complete robot retargeting optimizer, RL pipeline, or
reported experimental results.

## Overview

The main entry point is the `toporetarget` CLI. The code is organized around complete capabilities:

- canonical, robot-independent `HOISequence` data with scene-frame geometry and explicit SE(3)
  frame conversions;
- read-only inspection of one GRAB NPZ sequence and conversion to a canonical Zarr cache;
- explicit MANO semantic layouts and versioned MANO-to-MediaPipe21 mapping profiles;
- generic differentiable URDF hand FK, named qpos, target anchors, and Arti-MANO RH/LH inspection;
- a lazy GRAB index, native-time/native-mesh single-sequence adapter, contact modes, validation,
  provenance, and raw/canonical comparison;
- source/object/timestamp preservation reports and static or interactive geometry viewers;
- paper-fidelity auditing, assumptions tracking, and local Arti-MANO asset import support.

External datasets, MANO/SMPL-X models, robot assets, and extraction caches are not distributed with
this repository. Keep them outside Git under `.local/`-configured paths. The canonical data
interface is described in [`docs/HOI_DATA_INTERFACE.md`](docs/HOI_DATA_INTERFACE.md), and frame
semantics are defined in [`docs/COORDINATE_CONVENTIONS.md`](docs/COORDINATE_CONVENTIONS.md).

## TODO and roadmap

The complete staged TODO list is below. “Complete” means the bounded definition documented for
that stage; it does not imply full-dataset or result-level reproduction.

| Stage | Capability | Status | Definition of done / remaining TODO |
| ---: | --- | --- | --- |
| 0 | Repository architecture and path policy | Complete | CLI scaffold, configuration, dataset discovery, and Arti-MANO importer pass. |
| 1 | Paper fidelity audit | Complete | PDF manifest, equation/table/figure traceability, assumptions, and checker pass. |
| 2 | Canonical HOI schema and coordinates | Complete, bounded | Schema, lazy Zarr storage, comparison views, and bounded GRAB inspection pass. |
| 3 | MANO → MediaPipe-style 21 source adapter | Complete, bounded | Explicit layouts/profiles, converter, reports, viewers, synthetic tests, and bounded real GRAB checks pass; semantic and topology assumptions remain explicit. |
| 4 | Arti-MANO robot adapter | Complete, with assumptions | Generic URDF/FK interface, explicit MediaPipe-21-compatible anchors, separate geometry inspection, RH/LH validation, Jacobian checks, and CLI pass; paper frame/mapping assumptions remain explicit. |
| 5 | Full GRAB dataset adapter | Complete, bounded; fresh semantic closeout passed | Lazy index, native single-sequence/bimanual conversion, validation, provenance, raw/binary/official semantic contacts, and interactive HOI viewer; full-batch conversion remains out of scope. |
| 6 | Object sampling, collision geometry, and SDF | Complete, bounded; assumptions explicit | Mesh audit, deterministic 50-point surface references, collision-only robot samples, SDF queries, probes, reports, visualizations, and bounded real-data acceptance pass; later interaction/optimization remains out of scope. |
| 7 | Relative bone-direction initialization | TODO | Implement and test the paper's Eq. 1–2 initialization. |
| 8 | Interaction graph and Laplacian coordinates | TODO | Implement and test the Eq. 3–7 graph/deformation terms. |
| 9 | Constrained optimization with slack variables | TODO | Implement and test Eq. 8–9 constraints and optimization. |
| 10 | GRAB → Arti-MANO end-to-end retargeting | TODO | Produce a reproducible robot reference trajectory. |
| 11 | Metrics and ContactPose evaluation | TODO | Implement Eq. 10–12 metrics and report fixtures. |
| 12 | OakInk, DexYCB, and HO-Cap adapters | TODO | Add independently validated dataset adapters. |
| 13 | ARCTIC, OakInk2, and TACO extensions | TODO | Add independently validated dataset adapters. |
| 14 | Arbitrary dexterous-hand plugin interface | TODO | Test URDF/MJCF hand plugin contracts. |
| 15 | Baselines and ablations | TODO | Add fair OmniRetarget, Mink, DexPilot, and GeoRT runs. |
| 16 | Reference-tracking PPO | TODO | Add RL training and evaluation pipeline. |
| 17 | Paper experiment reproduction | TODO | Reproduce tables, figures, seeds, and result reports. |
| 18 | Performance optimization and v1.0 release | TODO | Establish benchmarks, packaging, and release criteria. |
| 19 | Non-paper extensions | TODO | Keep MANO cleanup, SPIDER, and other extensions separately labeled. |

The maintained roadmap with deliverables and status is [docs/ROADMAP.md](docs/ROADMAP.md); the
Chinese roadmap is [docs/ROADMAP.zh-CN.md](docs/ROADMAP.zh-CN.md).

## Quickstart

### Requirements

- Python 3.10–3.13
- Git
- External data/models only when using the corresponding workflows
- A graphical backend for `--show` viewers; `MPLBACKEND=Agg` is suitable for headless smoke tests

Install the complete environment for the currently implemented data and visualization workflows:

```bash
python -m pip install -e ".[dev,cache,viz,grab,robot,geometry]"
```

For core schema/tests without Zarr, visualization, or GRAB support, `python -m pip install -e ".[dev]"`
is sufficient.

### Configure local resources

Do not put datasets or model files in the repository. Use environment variables or the ignored
`.local/config.yaml`:

```bash
export GRAB_ROOT=/path/to/GRAB                 # contains grab/ and tools/object_meshes/
export MANO_MODEL_ROOT=/path/to/MANO/models    # contains MANO_LEFT.pkl/MANO_RIGHT.pkl
export MANIPTRANS_ROOT=/path/to/ManipTrans     # only needed for Arti-MANO import
```

The template is [`configs/paths.example.yaml`](configs/paths.example.yaml), and the data/license
boundary is [`docs/LICENSE_AND_DATA_POLICY.md`](docs/LICENSE_AND_DATA_POLICY.md).

### Check the installation

```bash
toporetarget --help
toporetarget data --help
toporetarget keypoints --help
toporetarget robots --help
toporetarget robots list
toporetarget doctor paper
```

## Workflows

The sections below are organized by complete user-facing capabilities rather than by development
stage. Each section starts with the core scripts/commands and then gives optional diagnostics.

### 1. Synthetic canonical HOI workflow

Create and inspect a deterministic canonical sequence:

```bash
toporetarget data make-synthetic \
  --output .local/cache/hoi/synthetic_demo.zarr \
  --num-frames 8

toporetarget data inspect \
  --input .local/cache/hoi/synthetic_demo.zarr \
  --frame 0

toporetarget data compare \
  --dataset synthetic \
  --sequence demo \
  --canonical .local/cache/hoi/synthetic_demo.zarr \
  --layout side-by-side \
  --frame 0 \
  --output .local/reports/stage2a/synthetic_side_by_side.png \
  --error-json .local/reports/stage2a/synthetic_side_by_side.json
```

Frame ranges are contiguous and half-open: `--start-frame 0 --end-frame 60` means frames 0–59.
The comparison `--show` mode is interactive; `--output` creates a headless image.

### 2. One GRAB NPZ to canonical Zarr

The historical Stage 2B reader is intentionally sequence-scoped. It reads one explicit NPZ and
selects one hand; the production Stage 5 adapter is documented below. For a full sequence, omit `--start-frame` and
`--end-frame`. For a bounded clip, provide both.

```bash
export GRAB_SEQUENCE="$GRAB_ROOT/grab/<subject>/<sequence>.npz"

toporetarget data describe \
  --dataset grab \
  --sequence-path "$GRAB_SEQUENCE" \
  --grab-root "$GRAB_ROOT" \
  --hand right \
  --mano-model-root "$MANO_MODEL_ROOT"

# Full trajectory: no --start-frame/--end-frame.
toporetarget data convert \
  --dataset grab \
  --sequence-path "$GRAB_SEQUENCE" \
  --grab-root "$GRAB_ROOT" \
  --hand right \
  --mano-model-root "$MANO_MODEL_ROOT" \
  --output .local/cache/hoi/grab/sequence_rh_full.zarr

# Optional bounded inspection: --end-frame is exclusive.
toporetarget data convert \
  --dataset grab \
  --sequence-path "$GRAB_SEQUENCE" \
  --grab-root "$GRAB_ROOT" \
  --hand right \
  --mano-model-root "$MANO_MODEL_ROOT" \
  --start-frame 0 \
  --end-frame 60 \
  --output .local/cache/hoi/grab/sequence_rh_f000000_f000060.zarr
```

The canonical cache contains the selected hand's MANO/source geometry, wrist pose, object state,
timestamps, and provenance. It is the input to the next workflow, not yet a MediaPipe21 cache.
Use [`docs/GRAB_INSPECTION.md`](docs/GRAB_INSPECTION.md) for raw/canonical comparison commands.

### 3. MANO source trajectory to MediaPipe21 trajectory

The Stage 3 converter consumes a canonical Zarr cache and writes a separate cache with an explicit
`mediapipe21` track. It performs named semantic mapping and explicit fingertip vertex mapping; it
does not mirror, resample, smooth, recenter, normalize, or modify the source track.

```bash
toporetarget keypoints layouts
toporetarget keypoints profiles
toporetarget keypoints describe-profile \
  --profile mano_v1_2_smplx_to_mediapipe21

toporetarget keypoints convert \
  --input .local/cache/hoi/grab/sequence_rh_full.zarr \
  --output .local/cache/hoi/grab/sequence_rh_full_mp21.zarr \
  --hand right \
  --mano-model-root "$MANO_MODEL_ROOT"

toporetarget keypoints validate \
  --input .local/cache/hoi/grab/sequence_rh_full_mp21.zarr \
  --hand right \
  --layout mediapipe21 \
  --report .local/reports/stage3/sequence_rh_full_validation.json \
  --csv .local/reports/stage3/sequence_rh_full_validation.csv
```

This CLI processes one selected hand at a time. Run the same two conversion commands with
`--hand left` for the left-hand trajectory. See [`docs/MANO_TO_MEDIAPIPE21.md`](docs/MANO_TO_MEDIAPIPE21.md)
for the mapping profile and assumptions.

### 4. Sequence visualization and debugging

Static PNG rendering:

```bash
toporetarget keypoints visualize \
  --input .local/cache/hoi/sequence_rh_full_mp21.zarr \
  --hand right \
  --layout mediapipe21 \
  --view scene \
  --frame 0 \
  --show-source-layout \
  --show-mesh \
  --show-labels \
  --output .local/reports/stage3/scene_first.png
```

Local interactive viewer:

```bash
toporetarget keypoints visualize \
  --input .local/cache/hoi/sequence_rh_full_mp21.zarr \
  --hand right \
  --layout mediapipe21 \
  --view scene \
  --start-frame 0 \
  --end-frame <num-frames> \
  --show \
  --show-source-layout \
  --show-mesh \
  --show-labels
```

The viewer provides a frame slider, previous/next buttons, scene/wrist switching, MANO mesh,
source MANO joints, MediaPipe21, skeleton edges, semantic labels, object mesh, and axes toggles.
It displays frame, timestamp, and mapping profile ID. Display transforms use temporary arrays and
do not change canonical keypoint coordinates. The detailed viewer contract is in
[`docs/MANO_TO_MEDIAPIPE21.md`](docs/MANO_TO_MEDIAPIPE21.md).

### 5. Production GRAB dataset adapter

Build a filename-first index, query it without loading frame arrays, and convert one right, left,
or bimanual sequence while retaining source timestamps, native meshes, personalized MANO `vtemp`,
object/table poses, and source, binary, or official semantic contacts:

```bash
toporetarget data index --dataset grab --output .local/index/grab
toporetarget data list --dataset grab --index .local/index/grab --subject s7 --limit 20
toporetarget data describe --dataset grab --index .local/index/grab \
  --sequence s7/cubemedium_inspect_1
toporetarget data convert --dataset grab --index .local/index/grab \
  --sequence s7/cubemedium_inspect_1 --hands both --start-frame 0 --end-frame 60 \
  --include-table --contact-mode source --include-mediapipe21 \
  --mano-model-root "$MANO_MODEL_ROOT" \
  --output .local/cache/hoi/grab/s7/cubemedium_inspect_1/both_f000000_f000060.zarr
toporetarget data validate --dataset grab --index .local/index/grab \
  --sequence s7/cubemedium_inspect_1 \
  --canonical .local/cache/hoi/grab/s7/cubemedium_inspect_1/both_f000000_f000060.zarr \
  --mano-model-root "$MANO_MODEL_ROOT" \
  --report .local/reports/stage5/grab_validation.json
```

Use `--contact-mode semantic` to retain the raw GRAB labels, derive the binary mask, and attach
the verified official 0--55 body/hand mapping from `configs/datasets/grab_contact_parts.yaml`:

```bash
toporetarget data convert --dataset grab --index .local/index/grab \
  --sequence s7/cubemedium_inspect_1 --hands both --start-frame 0 --end-frame 60 \
  --include-table --contact-mode semantic --include-mediapipe21 \
  --mano-model-root "$MANO_MODEL_ROOT" \
  --output .local/cache/hoi/grab/s7/cubemedium_inspect_1/semantic_f000000_f000060.zarr
```

The adapter has no temporal resampling, spatial/object surface sampling, raw-source writes, or
full-batch conversion. Use `toporetarget data visualize` for raw/canonical/compare modes, overlay
or side-by-side layouts, frame slider/keyboard playback, scene/object/wrist references, semantic
contact colors, and headless PNG output. The canonical CLI flag is `--reference-frame`; the older
`--reference` spelling remains a deprecated compatibility alias. See
[`docs/GRAB_DATASET_ADAPTER.md`](docs/GRAB_DATASET_ADAPTER.md) and
[`docs/GRAB_INTERACTIVE_VISUALIZATION.md`](docs/GRAB_INTERACTIVE_VISUALIZATION.md).

### 6. Arti-MANO asset import

Import only the local Arti-MANO asset tree from a separately checked-out ManipTrans source:

```bash
toporetarget assets import-artimano \
  --source-root "$MANIPTRANS_ROOT" \
  --destination .local/assets/artimano

toporetarget doctor assets
```

The importer records hashes and provenance in ignored local manifests. ManipTrans Python code is
not copied into this repository. See [`docs/UPSTREAM_REFERENCES.md`](docs/UPSTREAM_REFERENCES.md)
and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

### 7. Target Hand Asset Setup and Kinematic Inspection

The Stage 4 workflow uses the imported Arti-MANO assets as a target-hand model. Core inspection
commands are:

```bash
toporetarget robots list
toporetarget robots inspect \
  --robot artimano_rh \
  --json .local/reports/stage4/artimano_rh_inspect.json
toporetarget robots validate \
  --robot artimano_rh \
  --report .local/reports/stage4/artimano_rh_validation.json \
  --csv .local/reports/stage4/artimano_rh_validation.csv
toporetarget robots fk \
  --robot artimano_rh --pose neutral --dtype float64 \
  --output .local/reports/stage4/artimano_rh_neutral_fk.json
toporetarget robots anchors \
  --robot artimano_rh \
  --csv .local/reports/stage4/artimano_rh_anchors.csv
```

Run the same core commands with `artimano_lh` to load the actual left-hand URDF independently.
The registry list command does not require local assets; inspect and validation resolve the asset
root from `--asset-root`, `ARTIMANO_ASSET_ROOT`, `.local/config.yaml`, or the safe local default.

Debug/Inspection supplements after the core flow:

```bash
toporetarget robots jacobian-check \
  --robot artimano_rh --pose random --seed 4 --dtype float64 \
  --report .local/reports/stage4/artimano_rh_jacobian.json
toporetarget robots visualize \
  --robot artimano_rh --pose neutral --geometry visual \
  --show-keypoints --show-skeleton --show-labels --show-base-frame \
  --output .local/reports/stage4/artimano_rh_neutral_visual.png
toporetarget robots visualize \
  --robot artimano_rh --pose neutral --geometry collision \
  --show-keypoints --show-skeleton \
  --output .local/reports/stage4/artimano_rh_neutral_collision.png
toporetarget robots visualize \
  --robot artimano_rh --pose random --seed 4 --geometry both \
  --show-keypoints --show-skeleton --show-labels --show-joint-axes \
  --output .local/reports/stage4/artimano_rh_random_overlay.png
```

The interface reports missing collision geometry and does not synthesize it. It defines `palm` as
the engineering URDF base frame; it does not choose the paper's unresolved wrist-frame
parameterization or perform MANO-to-Arti-MANO retargeting. See
[`docs/ROBOT_HAND_INTERFACE.md`](docs/ROBOT_HAND_INTERFACE.md) and
[`docs/ARTIMANO_ADAPTER.md`](docs/ARTIMANO_ADAPTER.md).

### 8. Inspect Object Geometry, Generate Surface References, and Validate Signed Distance

This bounded geometry workflow keeps the existing canonical object-local mesh and Stage 4
collision geometry contracts. The paper fixes the object count at 50; the sampler, seed, temporal
reuse, normals, SDF backend, and robot collision count are explicit engineering assumptions.

```bash
toporetarget geometry inspect-mesh \
  --canonical .local/cache/hoi/grab/s7/cubemedium_inspect_1/right_f000000_f000060.zarr \
  --object-id primary --json .local/reports/stage6/grab_object_mesh_audit.json
toporetarget geometry sample-object \
  --canonical .local/cache/hoi/grab/s7/cubemedium_inspect_1/right_f000000_f000060.zarr \
  --object-id primary --profile paper_strict_area_uniform \
  --output .local/cache/geometry/object_surface/object.npz \
  --report .local/reports/stage6/object_samples.json
toporetarget geometry validate-samples --canonical .local/cache/hoi/grab/s7/cubemedium_inspect_1/right_f000000_f000060.zarr \
  --object-id primary --samples .local/cache/geometry/object_surface/object.npz \
  --report .local/reports/stage6/object_samples_validation.json
toporetarget geometry validate-sdf --shape sphere \
  --report .local/reports/stage6/sdf_sphere_validation.json
toporetarget geometry sample-robot --robot artimano_rh --pose neutral \
  --profile engineering_collision_32_per_geometry \
  --output .local/cache/geometry/robot_surface/artimano_rh.npz
toporetarget geometry probe-collision \
  --robot-samples .local/cache/geometry/robot_surface/artimano_rh.npz \
  --object-shape cube --report .local/reports/stage6/synthetic_collision_probe.json

# Visualize the fixed 50 object samples and their IDs/normals.
toporetarget geometry visualize-object \
  --canonical .local/cache/hoi/grab/s7/cubemedium_inspect_1/right_f000000_f000060.zarr \
  --object-id cubemedium \
  --samples .local/cache/geometry/object_surface/cubemedium_samples.npz \
  --frame 0 \
  --output .local/reports/stage6/object_samples_frame0_ids.png \
  --show-ids --show-normals --show-object-frame --show-scene-frame

# Repeat with --frame 29 and --frame 59 for middle/last-frame overlays.
```

The object viewer displays the fixed 50 sample IDs and normals; `--frame 29` and `--frame 59`
produce middle/last-frame overlays using the same face+barycentric identities and only changing
the object pose. Other debug visualizations include SDF slices and RH/LH collision surface samples.
See [`OBJECT_GEOMETRY_AND_SAMPLING.md`](docs/OBJECT_GEOMETRY_AND_SAMPLING.md),
[`SIGNED_DISTANCE_AND_COLLISION_QUERIES.md`](docs/SIGNED_DISTANCE_AND_COLLISION_QUERIES.md), and
[`stages/STAGE_6_OBJECT_GEOMETRY_SDF.md`](docs/stages/STAGE_6_OBJECT_GEOMETRY_SDF.md).

### 9. Paper traceability and reproduction audit

Run the repository-local paper audit and inspect the machine-readable fidelity configuration:

```bash
python scripts/check_paper_fidelity.py
toporetarget doctor paper
```

The audited PDF, equation/table/figure traceability, and unresolved assumptions are documented in
[`docs/PAPER_FIDELITY.md`](docs/PAPER_FIDELITY.md),
[`docs/PAPER_FIDELITY.yaml`](docs/PAPER_FIDELITY.yaml), and
[`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md).

### 10. Development validation

```bash
ruff check .
ruff format --check .
mypy src
pytest -q
python scripts/check_paper_fidelity.py
git diff --check
```

Licensed-data tests are opt-in and require local GRAB/MANO resources:

```bash
GRAB_SEQUENCE="$GRAB_SEQUENCE" \
MANO_MODEL_ROOT="$MANO_MODEL_ROOT" \
pytest -q tests/licensed_data
```

## Documentation map

- [Roadmap](docs/ROADMAP.md) / [中文路线图](docs/ROADMAP.zh-CN.md)
- [Canonical HOI interface](docs/HOI_DATA_INTERFACE.md)
- [Coordinate conventions](docs/COORDINATE_CONVENTIONS.md)
- [GRAB inspection](docs/GRAB_INSPECTION.md)
- [GRAB dataset adapter](docs/GRAB_DATASET_ADAPTER.md) / [interactive visualization](docs/GRAB_INTERACTIVE_VISUALIZATION.md)
- [MANO-to-MediaPipe21 adapter](docs/MANO_TO_MEDIAPIPE21.md)
- [Generic robot-hand interface](docs/ROBOT_HAND_INTERFACE.md)
- [Arti-MANO target adapter](docs/ARTIMANO_ADAPTER.md)
- [Stage 4 report](docs/stages/STAGE_4_ARTIMANO_TARGET_HAND.md)
- [Stage 5 report](docs/stages/STAGE_5_GRAB_DATASET_ADAPTER.md)
- [Object geometry and sampling](docs/OBJECT_GEOMETRY_AND_SAMPLING.md)
- [Signed distance and collision queries](docs/SIGNED_DISTANCE_AND_COLLISION_QUERIES.md)
- [Stage 6 report](docs/stages/STAGE_6_OBJECT_GEOMETRY_SDF.md)
- [Paper fidelity](docs/PAPER_FIDELITY.md)
- [Data and license policy](docs/LICENSE_AND_DATA_POLICY.md)
- [Development log](docs/DEVELOPMENT_LOG.md) / [中文开发日志](docs/DEVELOPMENT_LOG.zh-CN.md)
- [Contributing](CONTRIBUTING.md) / [third-party notices](THIRD_PARTY_NOTICES.md)

## License

The repository code and documentation are released under the GNU General Public License v3.0;
see [`LICENSE`](LICENSE). External GRAB, MANO/SMPL-X, ManipTrans, robot assets, and other datasets
remain subject to their own licenses and are not redistributed here. See
[`docs/LICENSE_AND_DATA_POLICY.md`](docs/LICENSE_AND_DATA_POLICY.md) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) before using external resources.

## Acknowledgments

This repository acknowledges:

- the authors of [*TopoRetarget: Interaction-Preserving Retargeting for Dexterous Manipulation*](https://toporetarget2026.github.io/TopoRetarget/);
- [the ManipTrans project](https://maniptrans.github.io/), whose local Arti-MANO asset tree is used only as an acquisition-side
  source;
- the GRAB dataset and the MANO/SMPL-X model ecosystem used by the bounded data workflows;

Please preserve upstream attribution and comply with each external project's terms when using
those resources.

## Citation

If this repository or its implementation notes are useful, cite the TopoRetarget paper:

```bibtex
@article{wu2026toporetarget,
  title   = {TopoRetarget: Interaction-Preserving Retargeting for Dexterous Manipulation},
  author  = {Wu, Jielin and Yao, Shenzhe and He, Guanqi and Liu, Xiaohan and Zeng, Zhaoqing
             and Jiang, Xiangrui and Yang, Han and Zhang, Wentao and Zhao, Hang},
  journal = {arXiv preprint arXiv:2606.16272},
  year    = {2026},
  doi     = {10.48550/arXiv.2606.16272}
}
```

Also cite GRAB, MANO/SMPL-X, and ManipTrans when using their data, models, or assets. The local
paper copy is [`docs/TopoRetarget.pdf`](docs/TopoRetarget.pdf), and upstream acquisition notes are
in [`docs/UPSTREAM_REFERENCES.md`](docs/UPSTREAM_REFERENCES.md).
