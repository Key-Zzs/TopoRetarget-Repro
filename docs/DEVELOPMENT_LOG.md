# Development log

This file preserves the former English README content as a chronological implementation snapshot.
For the user-facing repository overview, workflows, setup instructions, and project roadmap, see
the root [README](../README.md). The detailed reproduction record is in
[REPRODUCTION_LOG.md](REPRODUCTION_LOG.md).

## Repository status at the Stage 3 snapshot

- Stage 0 complete: repository scaffold, configuration, read-only dataset discovery, and local Arti-MANO importer.
- Stage 1 complete: complete 16-page paper audit, parameter provenance, assumptions, and fidelity checker.
- Stage 2A complete: canonical HOI schema, explicit coordinate semantics, opt-in Zarr storage,
  deterministic synthetic data, error metrics, and headless comparison visualization.
- Stage 2B complete for the bounded real-data acceptance: one GRAB sequence was reconstructed with
  the user-provided MANO models through the optional SMPL-X backend, converted to canonical Zarr,
  compared, and rendered at first/middle/last clip frames.
- Stage 3 complete for the bounded source-hand adapter: explicit MANO semantic mapping to
  MediaPipe-style 21 points, versioned profiles, dense/sparse regressor path, scene/wrist views,
  integrity reports, static and interactive viewers, synthetic tests, and real right/left-hand
  GRAB validation.
- Stage 4 complete with explicit assumptions: a generic YAML robot-hand spec/registry, strict
  URDF parser, differentiable Torch FK plus independent NumPy FK, named qpos and limits, canonical
  MediaPipe-21-compatible target anchors, separate visual/collision geometry instances, Jacobian
  checks, synthetic fixtures, and independently loaded Arti-MANO RH/LH validation.

This repository does not implement the TopoRetarget retargeting algorithm, MANO-to-robot qpos
conversion, numerical optimization, Delaunay/SDF, RL/PPO, or baselines. Stage 3 remains a
source-hand adapter, Stage 4 remains a target-hand kinematics interface, and Stage 5 remains a
bounded data adapter; none claims full retargeting or MediaPipe detector accuracy.

## Stage 4 implementation record

The target-hand contract is `P^r(q)` only. `palm` is the engineering URDF base frame and the
external scene base pose is passed as a homogeneous transform. The paper's exact wrist-centered
robot frame and base rotation parameterization remain `A_ROBOT_HAND_FRAME_001`.

The tracked RH/LH specs use 28 links, 27 joints, 22 actuated joints, 5 fixed joints, and an
explicit 22-name order audited against both imported URDFs and ManipTrans `artimano.py`. The
shared `artimano_mediapipe21` profile reuses Stage 3's semantic layout and uses link/joint origins;
multi-axis co-located joints and fixed fingertip joint origins are recorded under
`A_ROBOT_KEYPOINT_ANCHORS_001` and `A_ARTIMANO_KEYPOINT_MAPPING_001`.

The imported asset manifest was checked before loading. The local evidence is upstream commit
`a3d08cfe3c3a5868a7f057533bcaf759c5af4705`, 98 imported files, 64 valid mesh references, and
manifest SHA-256 `c8e2c885e95cf690ec362c45e10d77cd16a60d3760efa692856617f148fe212e`. Visual and
collision geometry stay separate; each side has 21 visual and 16 collision instances. Fixed tip
links are visual-only in this asset, so no collision replacement is generated
(`A_ARTIMANO_COLLISION_COVERAGE_001`).

Synthetic tests cover parser graph errors, all supported joint/geometry types, analytic FK,
batch/device/dtype behavior, base equivariance, named qpos, anchors, Jacobian finite differences,
geometry separation, registry loading, and validation. Opt-in local tests load both actual RH/LH
URDFs. The core commands are `toporetarget robots list|inspect|validate|fk|anchors|jacobian-check|visualize`.
Generated reports and PNGs belong under ignored `.local/reports/stage4/`; no asset file is tracked.

The next stage boundary is deliberately preserved: Stage 5 GRAB adapter, retargeting, bone
direction initialization, interaction geometry, collision queries, SDF, and PPO were not started.

The bounded GRAB reader, real acceptance command, and tolerance report are documented in
[`GRAB_INSPECTION.md`](GRAB_INSPECTION.md). This is one explicit 60-frame inspection, not a
full-dataset conversion.

## Stage 5 implementation record

Stage 5 adds a filename-first lazy GRAB index, `GrabDatasetAdapter`, source/binary contact modes,
optional MediaPipe21 derivation, personalized-vtemp MANO reconstruction, native object/table mesh
tracks, atomic Zarr caching, validation JSON/CSV, raw/canonical comparison, and an interactive
raw/canonical viewer. The accepted local dataset root was
the locally configured/discovered GRAB root; the index contains 1,335 active NPZ sequences across
subjects `s1`–`s10` and does not import MANO or frame arrays. The machine-specific root is retained
only in ignored `.local/reports/stage5/` evidence.

The real acceptance sequence was `s7/cubemedium_inspect_1`, 120 Hz, with right-hand and bimanual
clips `[0, 60)`. Native hand/object vertices, source timestamps, contacts, personalized `vtemp`,
and the GRAB row-vector object transform were preserved. Validation and raw/canonical comparison
passed at zero timestamp/translation/world-vertex error and approximately `1.71e-6` degrees
maximum rotation error. A legacy Stage 2B native-keypoint metric was unavailable because the old
cache lacks the formal native-keypoint field; it was reported as unavailable rather than inferred.

The interactive smoke test covered slider, callbacks, play/pause, reference changes, visibility
toggles, stable artists, and timer shutdown. Real native meshes use a viewer-only polygon fallback
for oversized meshes; canonical geometry is unchanged. Stage 6 and all later geometry,
retargeting, collision, SDF, and PPO work remain not started.

The viewer also implements display-only frame stride, playback-speed and source/hand/geometry
visibility controls, plus optional GIF/MP4 headless animation paths. A direct local Zarr store is
used for cache I/O so the standard Zarr format remains usable under the managed filesystem used
for this audit; display operations do not change canonical schema or source arrays.

## Data and local assets

The repository does not contain GRAB, OakInk, OakInk2, ContactPose, TACO, HO-Cap, ARCTIC, DexYCB,
MANO, or SMPL-X. Put external data under a local storage root using:

```text
<storage-root>/<registered-dataset-alias>/data/**
```

Machine-specific paths belong in ignored `.local/config.yaml` or environment variables. Start from
[`configs/paths.example.yaml`](../configs/paths.example.yaml) and [`.env.example`](../.env.example).

## Historical commands

```bash
python -m pip install -e ".[dev]"
toporetarget --help
toporetarget data --help
toporetarget data make-synthetic --output .local/cache/hoi/synthetic_demo.zarr
toporetarget data inspect --input .local/cache/hoi/synthetic_demo.zarr --frame 0
toporetarget keypoints layouts
toporetarget keypoints profiles
toporetarget keypoints validate --input .local/cache/hoi/grab/cubemedium_inspect_1_rh_f000000_f000060_mp21.zarr --hand right --report .local/reports/stage3/mapping_validation.json
toporetarget keypoints visualize --input .local/cache/hoi/grab/cubemedium_inspect_1_rh_f000000_f000060_mp21.zarr --hand right --view scene --frame 0 --show-source-layout --show-mesh --show-labels --output .local/reports/stage3/scene_mapping_first.png
toporetarget keypoints visualize --input .local/cache/hoi/grab/cubemedium_inspect_1_rh_f000000_f000060_mp21.zarr --hand right --layout mediapipe21 --view scene --start-frame 0 --end-frame 60 --show --show-source-layout --show-mesh --show-labels
toporetarget doctor datasets --root "$REF2DEX_STORAGE_ROOT" --max-depth 4
toporetarget assets import-artimano --source-root "$MANIPTRANS_ROOT" --destination .local/assets/artimano
toporetarget doctor assets
toporetarget doctor paper
toporetarget doctor all
```

The dataset doctor performs read-only, allowlisted, bounded directory discovery and ignores
unregistered storage directories. The Arti-MANO command imports the complete URDF/mesh tree from
ManipTrans into `.local/assets/artimano/`; the result is not tracked by Git. Paper traceability is
checked with `python scripts/check_paper_fidelity.py`.

## Historical development checks

```bash
ruff check .
ruff format --check .
mypy src
pytest -q
python scripts/check_paper_fidelity.py
git diff --check
```

See [`ROADMAP.md`](ROADMAP.md), [`PAPER_FIDELITY.md`](PAPER_FIDELITY.md), and
[`LICENSE_AND_DATA_POLICY.md`](LICENSE_AND_DATA_POLICY.md) for the contemporaneous project
boundaries. The canonical interface is documented in
[`HOI_DATA_INTERFACE.md`](HOI_DATA_INTERFACE.md) and coordinate semantics in
[`COORDINATE_CONVENTIONS.md`](COORDINATE_CONVENTIONS.md).
