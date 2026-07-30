# Development log

## 2026-07-29 -- Stage 11 contract freeze

Frozen the extension interfaces without adding a dataset or running a new
retargeting job. `toporetarget.contracts` now exposes Canonical HOI v2 with a
lossless v1 facade/migration, DatasetAdapter v1, RobotHandPlugin v1,
RobotReference v2, and MetricRegistry v1. GRAB is registered through the new
dataset facade; Arti-MANO and Wuji Hand2 Beta1 are exposed through the generic
robot plugin registry. Existing `data.*`, `robots.*`, and metric APIs remain
available.

RobotReference v2 supports NPZ/Zarr and validates qpos, base/object/link
coordinates, timestamps/FPS, joint order, robot hash, and dataset provenance.
Metric declarations distinguish `PAPER_EXACT`, `DATASET_PROXY`,
`GENERIC_GEOMETRIC`, and `ENGINEERING_DIAGNOSTIC`; proxy labels cannot claim
ground truth. A migration report is generated from an existing Stage 10
canonical cache/reference and writes only isolated `.local/reports/stage11_*`
outputs. Historical Stage 5–10 artifacts and retargeting numerics remain
unchanged. Stage 12–19 are now the forward roadmap.

## 2026-07-29 -- W2.2 Wuji continuity closeout

Completed the bounded W2.2 diagnostic closeout on `main`. Added explicit W2
q-step decomposition/attribution, fixed B0/B1/B2 isolated and operational
windows, deterministic synthetic routing, a real W3 five-frame shadow, static
HTML review, screenshots, performance/failure reports, and source/formal
artifact integrity checks. The closeout contains all 210 expected ablation
rows and 42 solver checkpoints under
`.local/experiments/wuji_hand2_continuous_v1/closeout_v1/`; formal baseline,
continuous, export, source, MANO, and historical Stage-10 artifacts remain
unchanged, as does the sibling `pene-loss` worktree.

W2's 13 absolute q-step transitions are all source/warm-driven, with zero
correction-driven or jump-and-return transitions. Formal W1/W2/W3 numerical,
collision, bounds, and continuity gates pass. The recommendation remains
`WUJI_CONTINUOUS_PROFILE_NOT_RECOMMENDED_WINDOW_FALLBACK_FAILED`: the real W3
shadow produces future hints but its joint SLSQP returns status 4 and its
center fails continuity; W3 penetration rate also regresses from 0.90 to
0.95. The ablation conclusion is preserved as
`ABLATION_INCONCLUSIVE_DUE_TO_SOLVER_FAILURE` because bounded B1/B2 rows fail.
No formal artifact was overwritten and no git add/commit/push/reset/clean/tag
was performed.

## 2026-07-28 -- W2.1 Wuji continuous correction repair

Added the separate `wuji_continuous_full_state_v1` engineering profile. It
audits and preserves the existing local seed-delta chart, transports the
previous accepted correction through the current warm state, adds normalized
base/finger correction temporal energy, separates single-frame feasibility
from trajectory continuity, and records deterministic retry/window metadata.
The historical v3 profile and artifacts remain unchanged; no post-filtering or
paper-core objective change is permitted.

## 2026-07-27 -- F0 tracked robot assets and generic target-hand foundation

Completed F0 on `main`. Arti-MANO is now a tracked vendor snapshot at
`third_party/robot_hands/artimano/`, pinned to ManipTrans commit
`a3d08cfe3c3a5868a7f057533bcaf759c5af4705`. The snapshot contains the two URDFs and 96 mesh
files plus `LICENSE`, `SOURCE.yaml`, `NOTICE.md`, and a hash manifest; no ManipTrans Python source
was copied. The source manifest hash is
`1d14cce93e2ee09dedbfcda842b1d8aac29443f86b57a0a15f6289bd55e0f771`.

Extended the data-driven target-hand contract with asset bundle, kinematic, semantic-anchor,
surface, collision, and simulation specs. `RobotHandRegistry` is now the construction path for
Arti-MANO RH/LH; the legacy `load_artimano_model` API delegates to it. Resolution records tracked,
explicit override, or deprecated legacy fallback, and the new `robots resolve-assets` and
`compare-assets` commands expose provenance and migration state. Anchor/Jacobian shape logic is
profile-driven rather than assuming every hand has 21 points or 22 DoFs.

The F0 audit compared the tracked and pre-existing local payloads and found exact topology, qpos
order, limits, FK, anchors, Jacobians, mesh transforms, and source bytes for both sides. Historical
Stage 7–10 reports/exports remain read-only and rebind through source hashes; no solver was invoked
and no historical artifact was rewritten. Evidence is under `.local/reports/f0/` and remains
untracked. Wuji Hand2, `develop/pene-loss`, SDF penetration loss, solver-profile changes, and RL
remain outside F0.

## 2026-07-27 -- W0/W1 Wuji Hand2 Beta1 generic target integration

Completed W0/W1 on `main`. The approved Hand2 Beta1 body subset from
`wuji-technology/wuji-description` is tracked under `third_party/robot_hands/wuji_hand2_beta1/`.
The requested `release/v2026.7.23` ref resolves to commit
`2b57d2621caed4e65207bb767ba25fc8eaec0881`; the same-version tag is a different commit and was not
substituted. `SOURCE.yaml` records the MIT license, exclusions, import tool version, per-file hashes,
and source manifest hash.

Added independent RH/LH generic specs: 26 links, 25 joints, 20 actuated joints, 5 fixed joints,
roots `r_wrist`/`l_wrist`, explicit 20-DoF qpos/URDF/MJCF/actuator orders, MediaPipe-21 anchors,
surface policies, and separate URDF/MJCF collision profiles. Backend-free consistency checks pass
for neutral/midpoint/random qpos, including axes, limits, root, tip sites, link transforms, anchors,
mesh references, and ten MJCF contact-exclude pairs. Soft-pad tip meshes remain visual payloads and
are not silently promoted into formal collision.

The generic Stage 7/8 paths were widened from historical 22-DoF assumptions. A bounded airplane
window `[240,243)` passes warm-start validation, source-only graph/evaluation, collision QuerySet
construction, and Stage 9 objective/constraint/Jacobian construction. The smoke reports
`optimization_performed=false`; W2 full multi-clip Wuji retargeting is not claimed. The
`develop/pene-loss` worktree, upstream checkout, canonical source, object samples, and historical
artifacts remained unchanged.

## 2026-07-24 -- GRAB Arti-MANO quality A–E

Implemented the frozen G1–G4 quality experiment. Stage A binds the prescribed
native ranges and right-hand identity, reuses matching airplane artifacts, and
executes the retained v2/v3 solver profiles with checkpoint/resume. Stage B
adds the generic visual surface contact contract and deterministic Arti-MANO
regions. Stage C adds seed-only morphology candidates and fixed normalized-prior
diagnostics. Stage D declares the complete fixed contact grid and records proxy
gates without weakening collision constraints. Stage E records E0–E3 lineage,
Pareto gates, baseline fallback, reports, and four self-contained HTML viewers.

All new outputs are under `.local/experiments/grab_artimano_quality_v1/`;
raw source and historical Stage 5–10 artifacts remain read-only and `.local`
remains untracked. ContactPose is deferred and the result scope is limited to
the `s1` within-subject multi-object development benchmark.

The first real A-stage execution exposed a generic active-set continuation
round-off defect: a new soft constraint could start approximately `2.7e-8 m`
infeasible after reference/solver SDF conversion. A fixed `1e-9 m`
initialization-only interior buffer was added without changing the objective,
solver YAML, tolerance, QuerySet, active margin, or acceptance gate. The open
banana mesh was then handled by the authorized
`hybrid_original_distance_proxy_sign_v1` contract: original distance and
visual/contact semantics remain source-mesh based, while the derived proxy is
used only for sign. The proxy passes its fixed geometry gates, but the first
active QuerySet still contains three samples in the original boundary
exclusion zone (two on synthetic patch faces). The run therefore stops at G3
with `hard_blocker=SIGN_PROXY_CONTACT_REGION_CONFLICT`; no margin, frame,
profile, raw mesh, or G4/C–E continuation was changed. The recorded status is
`GRAB_QUALITY_A_TO_E_BLOCKED`.

## Q1–Q3 Multi-Dataset Interaction Benchmark (2026-07-24)

The benchmark implementation adds a versioned `toporetarget.hoi_benchmark.v1` selection
contract, a lazy GRAB selector, a ContactPose directory/schema adapter, a separated metric
registry, Eq. (10)–(12) implementations, GRAB contact proxies, dynamic/static applicability,
manifest-bound baseline execution, automatic gates, macro aggregation, and a self-contained HTML
dashboard. Selection is written and locked before baseline execution; source data and official
artifacts remain read-only. Existing `s1/airplane_lift`, right hand, global `[240,300)` is retained
as the fixed unit. The new stage does not change Eq. (1)–(9), introduce per-unit tuning, or claim
author-exact Eq. (9) semantics.

Read-only execution evidence for this checkout: the required GRAB/ContactPose/MANO/Arti-MANO
paths and explicit `topo-retarget` imports passed preflight. The existing GRAB index contains 1,334
non-fixed entries; a bounded 16-entry native-contact probe selected `s1/apple_eat_1 [212,272)`,
`s1/banana_lift [1658,1718)`, and `s1/alarmclock_lift [407,467)` alongside the fixed
`s1/airplane_lift [240,300)`. ContactPose inspection found 110 annotation candidates, 0 selected,
and 110 `official_contact_annotation_unavailable_or_unrecognized` rejections (12 also carry the
diagnostic deep-concave exclusion reason). The recorded status is
`Q1_CONTACTPOSE_SELECTION_BLOCKED`; selection was not frozen and no baseline/evaluation was run.

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
for oversized meshes; canonical geometry is unchanged. In that Stage 5 snapshot, Stage 6 and all
later geometry, retargeting, collision, SDF, and PPO work remained not started.

The viewer also implements display-only frame stride, playback-speed and source/hand/geometry
visibility controls, plus optional GIF/MP4 headless animation paths. A direct local Zarr store is
used for cache I/O so the standard Zarr format remains usable under the managed filesystem used
for this audit; display operations do not change canonical schema or source arrays.

## Stage 5 semantic-contact and CLI closeout

The contact contract was closed against the official `otaheri/GRAB/tools/utils.py` `contact_ids`
table at commit `4dab3211fae4fc5b8eb6ab86246ccc3a42d8f611` (source SHA-256
`bbdae13c1c437d60d22e2e8eabbabb7c2282a47918735876383794739d38a4a7`). The tracked mapping covers
labels `0..55`, keeps `0` as no-contact, preserves raw labels, derives `binary = labels != 0`,
and stores official integer semantic IDs plus a versioned mapping table. Strict mode fails on
unmapped labels; non-strict mode uses explicit ID `56` and records the loss.

The visualization CLI now documents `--reference-frame`, keeps `--reference` as a deprecated alias
with a warning, accepts equal duplicate values, and rejects conflicting values. The viewer can
switch source/binary/semantic contact colors and reports mapping identity in the legend. Current
closeout reports and semantic-enriched real caches are under ignored `.local/reports/stage5_closeout/`
and `.local/cache/`.

Fresh MANO-backed semantic conversion and validation now pass for the bounded real clips when the
external MANO root is supplied explicitly. The s1 contact window reports labels `[0,43,46,55]`
with no unmapped values and exact raw/binary/semantic/mapping round trips; the s7 bimanual geometry
window validates both hands and the table. The external MANO files remain runtime inputs and were
not copied into the repository.

## Stage 6 object geometry, deterministic sampling, and signed distance

Stage 6 reuses `MeshDefinition`/`RigidObjectTrack`, the existing SE(3) helpers, and the Stage 4
collision-geometry/FK API. A read-only mesh audit records source and derived hashes, topology,
watertightness, winding, degenerate faces, and sign reliability without repairing source data.

The paper-locked object count is loaded from `configs/paper/retarget.yaml` and resolves to 50. The
engineering profile `paper_strict_area_uniform` uses area-weighted triangle selection, square-root
barycentric coordinates, and an explicit NumPy PCG64 seed `20260720`. Face indices and barycentric
coordinates are retained so samples reconstruct exactly after scale changes. Anchors are sampled
once in the object frame and transformed per frame; they are not resampled and no FPS profile is
claimed as paper-exact. Normals are diagnostic face normals only. These unpublished choices are
tracked as `A_OBJECT_SAMPLING_001`, `A_OBJECT_SAMPLING_METHOD_001`,
`A_OBJECT_SAMPLING_SEED_001`, `A_OBJECT_SAMPLE_TEMPORAL_REUSE_001`, and
`A_SURFACE_NORMAL_MODE_001`.

The SDF foundation uses chunked analytic point-to-triangle closest points and generalized winding
solid angles. `strict` rejects open/non-manifold meshes, `winding` exposes confidence and ambiguity,
and `unsigned_only` makes signed distance unavailable instead of fabricating a positive sign. The
repository convention is positive outside. Scene queries transform points into object-local space,
then transform closest points and normals back using the existing rigid-frame helpers. Edge/vertex
closest points are marked non-smooth; the local linearization stops at geometric quantities and does
not create a q-space Jacobian.

Robot samples use only `collision_geometry_instances()` from Stage 4. The explicit engineering
profile is 32 samples per geometry, not a paper value. RH/LH Arti-MANO each expose 16 collision
geometries and 512 samples; visual-only tip links are reported, not replaced. Pointwise collision
probe output includes link/geometry/sample identity, sign confidence, and penetration depth, but no
final `Q_t`, Delaunay, Laplacian, slack, or optimization.

Bounded acceptance used `s7/cubemedium_inspect_1` frames `[0,60)` and local RH/LH assets. Reports and
images are under ignored `.local/reports/stage6/`; derived sample caches are under
`.local/cache/geometry/`. Source NPZ, mesh, canonical cache, MANO, and Arti-MANO asset hashes remain
unchanged. Stage 7 was complete with explicit assumptions at this pre-Stage-8 snapshot; the
subsequent Stage 8 closeout is recorded below.

## Stage 7 — relative bone-direction warm starts (2026-07-20)

Stage 6 closeout was verified before editing: commit `8c5b1c7`, clean index/worktree, and bilingual
Stage 6 documentation complete. Stage 7 uses the existing canonical `mediapipe21` scene track, the
Stage 4 differentiable Arti-MANO FK/anchors, and a separate warm-start Zarr artifact. It does not
consume Stage 6 object samples, SDF values, Delaunay, Laplacian, collision-query, or PPO modules.

The selected default frame profile is `canonical_keypoint_wrist_v1`. Its wrist origin, middle-MCP
longitudinal axis, Gram-Schmidt index-minus-pinky lateral axis, and cross-product third axis are
shared by source and robot. A translation-centered scene-axis profile is retained for a bounded
observability comparison. Both profiles reject explicit degeneracies in strict mode and preserve
RH/LH semantic ordering; stored GRAB wrist poses are not equated with the Arti-MANO palm frame.

The default semantic bone profile has five full chains, 20 directed bones, and 15 consecutive
within-finger pairs. A phalange-only diagnostic has 15 bones and 10 pairs. Unit directions and
un-normalized adjacent differences are differentiable and batch-capable. Eq. (1) reports an exact
sum, not a mean, angle loss, absolute-direction loss, or bone-length-weighted loss.

Eq. (2) uses raw 22-joint radians. `lambda_warm=1` and `lambda_smooth=2.5` are read from the paper
config. Frame zero starts at neutral q with no temporal residual; subsequent frames use only the
previous successful warm-start q. SciPy TRF with direct URDF bounds and a Torch-autograd float64
Jacobian is the explicit engineering solver. Paper objective values are recomputed instead of
copying SciPy's half-cost. Strict failures stop with a frame/status message.

The local direction objective makes base translation unobservable and, for the default local
profile, removes base rotation observability as well. The implementation therefore optimizes q_theta
only and records qpos Jacobian singular values/rank plus synthetic base Jacobians. After solving,
the explicit non-paper base seed is `T^S_B=T^S_Hs(T^B_Hr(q))^-1`; alignment errors are stored in
the artifact and validation report.

The bounded real acceptance used `s7/cubemedium_inspect_1`, `[0,60)`, native 120 FPS, and local
Arti-MANO RH/LH assets. RH input was `cubemedium_inspect_1_rh_f000000_f000060_mp21.zarr`; LH used
the existing Stage 5 semantic left-hand clip `semantic_left_f000000_f000060.zarr`. Both produced
60-frame `toporetarget.warm_start.v1` artifacts with 22-D qpos, successful bound-constrained solves,
non-increasing per-frame total objective, exact FK/base-frame alignment, and source-cache hash
matches. The artifacts and report/images are under ignored `.local/` paths.

Tests added coverage for 20/15 and 15/10 topology, RH/LH frame semantics, rigid invariance,
translation-centered diagnostics, strict degeneracy, exact Eq. (1) sum, Torch float32/float64
autograd, Eq. (2) residual scaling, base non-observability, and artifact round-trip. Stage 7 is
`implemented_with_assumptions`; Stage 8 was the next bounded closeout.

## Stage 8 — source interaction graphs and Laplacian loss (2026-07-20)

Stage 8 adds a separate source-only interaction graph artifact and a frozen Eq. (7)
evaluation artifact. Each bounded RH/LH clip uses 60 frames, 21 canonical MediaPipe-21
source points, and the fixed 50-point Stage 6 object sample artifact. One explicit
non-incremental SciPy/Qhull Delaunay call is made per source frame with `Qbb Qc Qz Q12`;
its unique tetrahedron edges and source-derived directed weights are reused by robot FK.
The strict profile uses centroid/bounding-box-diagonal conditioning only for Qhull, while
all source vertices, volumes, distances, and weights remain in metres.

Eq. (6) is implemented with a differentiable Torch sparse scatter Laplacian and Eq. (7)
is the exact mean squared residual divided by 71. Evaluation reads Stage 7 qpos/base,
keeps them unchanged, preserves object point identity, emits qpos Jacobians and bounded
base diagnostics, and records zero robot-side Delaunay, optimization, SDF, and collision
access. Eq. (8)-(9), slack, collision constraints, and RL remain unimplemented.

Real RH/LH graph/evaluation validation, identity and scaled-residual oracles, topology
over-time, object-scale diagnostics, input/source-integrity audits, unit tests, and static
plus interactive visualization smoke tests are stored under ignored `.local/` paths. The
Stage 8 status is `implemented_with_assumptions`; no Stage 6/7 artifact or source hash was
modified, and no git commit/push/tag was performed.

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

## Stage 9 — final constrained interaction-preserving refinement (2026-07-20)

Stage 9 implements the bounded Eq. (8)-(9) refinement on the existing RH/LH
`s7/cubemedium_inspect_1` 60-frame inputs. The solver consumes the frozen Stage 7 warm start,
Stage 8 graph, Stage 6 cubemedium mesh, and immutable 512-point robot collision surface. It uses
explicit local seed-delta coordinates, previous-final remapping, paper weights from
`configs/paper/retarget.yaml`, positive-outside reference SDF constraints, per-query slack, and
independent full-surface audits.

Full and adaptive QuerySet profiles are deterministic and hash-recorded. The adaptive profile
starts with penetration/10 mm/nearest-per-geometry samples and monotonically adds full-surface
violations. The solver is float64 SLSQP with Torch-autograd objective and hybrid SDF constraint
Jacobians. A convex-hull solver backend is used only after probe comparison to the Stage 6
reference backend; acceptance always uses the reference backend.

Implementation issues closed during this stage were zero-angle SO(3) gradient NaNs, incorrect
geometry-sample slicing, strict reference-vs-solver SDF separation, and async Zarr array
creation/loading. The engineering profile records `maxiter=30`, `ftol=1e-7`, and fail-fast status
handling because the paper does not disclose optimizer details. Stage 9 remains
`implemented_with_assumptions`; Stage 10, RL, physics, ContactPose, and baseline behavior were
not started. All generated outputs and pre-stage snapshots remain under ignored `.local/`; no git
add/commit/push was performed.

The final closeout reran the complete `[0,60)` range for both hands. RH reached minimum full
signed distance `0.623582905 m`, zero penetration, maximum slack `2.137e-6 m`, mean/p95 solve
time `20.146/22.435 s`; LH reached `0.641271031 m`, zero penetration, maximum slack
`5.096e-7 m`, mean/p95 `19.214/20.853 s`. Both independent validation reports passed, and
all 47 non-time arrays matched exactly between the original and full rerun artifacts for each
hand. At frames `0/29/59`, adaptive used 16 queries versus 512 for full reference; the largest
observed differences were `8.20e-6 m` in minimum full SDF and `8.77e-10` in objective value.
Jacobian checks passed for RH/LH with constraint max errors below `2.03e-10` and no finite-difference
fallbacks. Reports are under `.local/reports/stage9/`; the known canonical `metadata.json` Zarr
sidecar warning is pre-existing and does not modify source artifacts.

## Stage 10 — bounded workflow orchestration (2026-07-20)

Stage 10 adds a manifest-driven 19-node GRAB-to-Arti-MANO DAG, official semantic contact-window
selection, content-addressed cache/resume/invalidation, source-integrity snapshots, semantic sanity
and cross-stage identity reports, artifact-only review rendering, and read-only `robot_reference.v1`
export. New workflow configuration is under `configs/workflows/`; user-facing procedures are in
`docs/END_TO_END_GRAB_ARTIMANO.md`, `docs/WORKFLOW_RESUME_AND_PROVENANCE.md`, and
`docs/TRAJECTORY_VISUALIZATION.md`.

The selector passed on `s1/airplane_lift` windows `[844,904)`, `[240,300)`, and the specified
`[238,298)`, and on the existing Stage 9 object `s7/cubemedium_inspect_1` at `[363,423)`, with
official right-hand semantic contact and strict watertight object geometry. Additional finite,
explicitly queried candidates included `s1/airplane_fly_1 [729,789)`, `s1/cubemedium_inspect_1
[343,403)`, and the ratio-0.5 transition window `s1/airplane_fly_1 [159,219)`. The small-cube
candidate `[984,1044)` was rejected by the unchanged strict Stage 8 graph at frame 13 because two
simplex volumes were below tolerance. The other completed contact-rich runs reached frozen
interaction evaluation, then stopped at the unchanged Stage 9 SLSQP refinement with `Iteration
limit reached` (frame 0 or 1). The transition run exceeded the normal bounded runtime, was stopped
after the solver child reached roughly 100% CPU for more than 40 minutes, and was recorded as a
SIGTERM failure rather than a success. No Stage 7–9 solver, weight, coordinate, or threshold was
changed to bypass any result. Stage 10 is therefore implemented with real acceptance blocked at
the existing contact-rich refinement convergence boundary; per-run input, reuse, performance,
determinism-pending, semantic, source-integrity, and summary reports are retained under ignored
`.local/runs/stage10/`; no commit, tag, or push was performed.

A finite follow-up pass queried explicit `s1/apple_lift`, `s1/cylinderlarge_inspect_1`,
`s1/spheremedium_inspect_1`, `s1/mug_lift`, `s1/phone_lift`, and
`s1/stanfordbunny_inspect_1` sequences. Apple and cylinderlarge passed strict selection;
sphere failed unchanged Stage 8 graph validation, while mug, phone, and stanfordbunny
were rejected for non-watertight meshes. The new `cylinderlarge_inspect_1 [327,387)`
run again passed Stage 8 and failed at Stage 9 frame 0 with `Iteration limit reached`.
A read-only one-frame diagnostic on `airplane_lift [240,300)` recorded SLSQP status 9
at frozen `maxiter=30`, although the returned candidate had full-surface minimum signed
distance `+0.01184 m` and positive hard/soft residual minima. This confirms the existing
strict fail-fast solver boundary; Stage 10 does not relax it. The diagnostic is retained
at `.local/reports/stage10/contact_rich_solver_diagnostic.json`.

The corresponding finite left-hand query for `s7/cubemedium_inspect_1 [513,573)` passed
contact and strict mesh selection but failed the unchanged Stage 8 graph at frame 1 due to
one simplex volume at or below `1e-24`; it did not enter final refinement.

## Stage 9.1 solver-robustness closeout (2026-07-21)

Stage 9.1 preserves the v1 SLSQP profile and adds the independent contact-rich
v2 profile. The active-set bug was that an expanded QuerySet rebuilt its
initial vector from the Stage 7 warm seed. v2 now continues from the prior
`result.x`, copies base/q coordinates, remaps old slack by query ID, and uses
the minimum bounded slack formula for new IDs. Query-set growth is monotonic
and the continuation trace is part of artifact provenance.

Termination is now decomposed into optimizer status/counters and independent
primal, bounds, active-set, full-surface hard/soft, finite-value, and acceptance
fields. A feasible status-9 result remains rejected by the strict policy;
`feasible_stationary_v1` is deferred and was not enabled. The fixed benchmark
grid is authoritative in `.local/reports/stage9_1/maxiter_benchmark.json`:
35 records over `[30, 60, 100, 200, 400]` select the minimum uniform budget
`100`. The preserved v1 profile hash is
`6affff2fdb425a0402f643c291c0b8904d4dbec6c5b69a5006cf9829dcc220aa`; the v2
profile hash is `c42c21d894c54d07b1d30943b5a3338b13628bf0429ab203b5540cf934d09b7c`.
The fixed-window benchmark passed at 100, but the opt-in full 60-frame
contact-rich artifact and deterministic repeat were not produced within the
bounded runtime window, so Stage 10 remains blocked pending that run. Window
geometry and the far-vs-contact comparison are recorded in
`.local/reports/stage9_solver_closeout/`; this is not a status-9 relaxation.
The Stage 10 resume plan selects v2 explicitly, invalidates Stage 9 and
downstream signatures only, and reuses Stage 5-8 artifacts. Solver and
termination remain paper-undisclosed implementation assumptions.
The bounded v2 rerun was then paused for the Stage 9.2 performance and
recoverability phase: the complete sequence is still performance-blocked.
This preserves the tested closeout changes but does not claim Stage 9.1
complete, a 60-frame artifact, or a deterministic repeat.

## Stage 9.2 refinement performance and recoverability (2026-07-21)

Stage 9.2 adds an execution layer around the frozen Stage 9 math: immutable
per-frame context, exact float64 x/query cache invalidation, persistent mesh and
SDF resources, batched collision-point Jacobians, explicit full-512 audit
scheduling, atomic strict-accepted frame checkpoints, soft wall-time pause,
resume, assembly, and fresh/resumed comparison commands. The solver profile,
paper weights, signed-distance convention, 512 samples, v2 continuation, and
strict status-9 rejection remain unchanged.

The CLI now exposes `profile-refinement`, `checkpoint-status`,
`validate-checkpoints`, `assemble-refinement`, and `compare-refinement-runs`.
The execution profile is CPU float64 `cached_checkpoint_cpu_float64_v1`,
separate from the SLSQP profile. Focused tests pass; the complete contact-rich
60-frame run, deterministic fresh/resumed repeat, and runtime-gate decision
remain evidence tasks. Until those reports exist, Stage 10 is not unblocked.

## Stage 10 accepted reference-runtime milestone (2026-07-21)

The user-accepted `s1/airplane_lift` right-hand window `[240,300)` is now
materialized under `.local/runs/stage10_reference_runtime/`. The new manifest
references the accepted Stage 9.2 final artifact, records all Stage 5–9 nodes
as reused, and records zero Stage 9 solver invocations. Cross-stage identity,
semantic sanity, NPZ/direct-Zarr round trips, and static viewer smoke pass.
The preferred performance gate remains false and performance debt remains open.
This accepted bounded reference-runtime milestone supersedes the earlier
pre-v3 Stage 10 blocked-run summaries; those entries remain as historical
failure evidence for the old contact-rich windows.
See [`REFINEMENT_PERFORMANCE.md`](REFINEMENT_PERFORMANCE.md),
[`REFINEMENT_CHECKPOINT_AND_RESUME.md`](REFINEMENT_CHECKPOINT_AND_RESUME.md),
and [`stages/STAGE_9_2_REFINEMENT_PERFORMANCE.md`](stages/STAGE_9_2_REFINEMENT_PERFORMANCE.md).

The bounded full run subsequently completed all 60 contact-rich frames in
`1075.941 s` (`17.932 min`) of solver compute. This earlier v1 evidence is
superseded by the v3 optimization run below; its strict-acceptance and recovery
history remains preserved in the reports.

The v3 execution profile uses analytic URDF spatial Jacobians, strict reference
recovery, and SDF tree leaf size 512. The first run completed 60/60 frames with
median `10.766 s`, p95 `38.711 s`, and `1104.827 s` total solve time. Its
deterministic repeat completed 60/60 with median `10.773 s`, p95 `39.052 s`, and
`1107.368 s` total. Both runs have status-0 strict acceptance, valid checkpoint
chains, and independent `60 x 512` reference validation; the maximum signed-
distance error is `2.50e-16 m`. All persisted arrays compare exactly after
excluding `solve_time_s` and documented metadata. The final status remains
`STAGE9_2_COMPLETE_REFERENCE_RUNTIME`; the preferred single-frame gate is not
met, while the explicitly accepted reference-runtime Stage 10 milestone is
complete.

## Stage 10.x interaction-mesh HTML visualization (2026-07-21)

Extended the manifest-driven mesh viewer with five switchable modes: `mesh`,
`full-graph`, `figure4-style`, `laplacian-diagnostic`, and `combined`. The page
keeps source/warm-start/final meshes visible in every mode, reuses the accepted Stage 8 graph/evaluation artifacts, and preserves the 21 hand +
50 object vertex contract and frozen directed weights, and computes final
Laplacian residuals in memory for diagnostics only. It adds edge category,
threshold/top-k, residual target/scope, scalar/vector, labels, and state-layer
controls. No solver, graph rebuild, or input-artifact write is performed.

The accepted `s1/airplane_lift` right-hand `[240,300)` run generated all five
HTML variants and passed headless Chrome smoke checks. Artifact content hashes
and mtimes were unchanged across generation. The viewer is an inspection aid;
formal interaction, collision, continuity, bounds, solver, and provenance gates
remain authoritative. No git add, commit, or push was performed.

## Stage 9.3 contact-retention and collision-geometry audit (2026-07-22)

Implemented the manifest-driven audit workflow and self-contained HTML review
for the accepted `s1/airplane_lift` right-hand `[240,300)` reference runtime.
The full 60-frame run uses deterministic dense surface samples, explicit
positive-outside signed-distance provenance, source/warm/final contact proxies,
visual-vs-collision offsets, QuerySet per-point/per-link reports, same-definition
Stage 9 objective comparisons, and a non-optimizing warm-to-final interpolation
diagnostic. All formal inputs retained their hashes and mtimes; solver
invocation count was zero. Contact retention and physical trackability remain
diagnostic/inconclusive or unverified where the available artifacts do not
support a stronger claim. The optional shadow evidence was not run and is
recorded as missing. No git add, commit, push, reset, or tag was performed.

## Stage 9.3.1 signed-distance reconciliation and bounded shadow gate (2026-07-22)

Added a read-only reconciliation workflow and CLI for the accepted `s1/airplane_lift`
right-hand `[240,300)` reference runtime. It proves the Stage 9.2 persisted 512-point
identity/order and transform chain, replays the formal acceptance contract independently
at `60/60` with zero mismatches, and records the signed-distance definition matrix.
The persisted values match the unified reference triangle/winding backend to machine
precision. The legacy Stage 9.3 `convex_hull_exact_solver_only` values do not match that
definition, so the unique state is `RETURN_TO_STAGE9_2_ACCEPTANCE_OR_METRIC_FIX`.

The directional offset audit correctly leaves the existing unsigned-only inflation label
inconclusive because the visual meshes are open. The requested shadow boundary selected
three frames but ran zero profiles and invoked zero solvers because the reconciliation gate
failed. Formal Stage 9.2/Stage 10 artifacts, exports, and manual acceptance were unchanged;
all changes remain unstaged.

## Stage 9.3.2 canonical SDF re-audit (2026-07-22)

Added the versioned `reference_winding_v1` formal contact-distance contract,
v2 canonical re-audit workflow, legacy-vs-canonical disagreement reports,
source/warm/final proxy metrics, collision/visual coverage diagnostics,
readiness reporting, and an isolated six-profile bounded shadow boundary.
Formal evaluation is separate from the approved Stage 9.2 solver backend;
legacy convex-hull values are diagnostic-only and superseded for formal
contact claims. The workflow preserves Eq. (1)-(9), paper weights, all
accepted Stage 9.2/Stage 10 artifacts, manual acceptance, and robot export.
Audit-only solver invocation remains zero; shadow outputs, if the gate passes,
are diagnostic and isolated. All code changes remain unstaged.

## Stage 7.1 warm-start fidelity and reachability audit (2026-07-23)

Added the manifest-driven, read-only `workflow audit-warm-start` boundary and
its HTML/report contract. The accepted `s1/airplane_lift` RH reference runtime
(`local [0,60)`, global `[240,300)`) replays all persisted Stage 7 qpos and
passes Eq. (1)/(2), frame, base, source mapping, and robot mapping gates at
machine precision. The audit records thumb URDF ancestry/axes, joint-limit
margins, Jacobian projection, Kabsch alignment alternatives, raw versus
robot-length thumb targets, and warm/final formal/keypoint/E_IM/contact
attribution.

Five bounded frames ran diagnostic-only IK and 4096-point Sobol workspace
sampling. Raw thumb targets averaged about 12.51 mm from the sampled workspace,
whereas robot-length targets averaged about 3.81 mm and were near it on every
selected frame; this is recorded as a morphology/embodiment gap, not a change to
the formal Stage 7 objective. Whole-hand final keypoint RMSE and E_IM increased
relative to warm, so final refinement degradation remains a separate ranked
factor. The final readiness is
`WARM_START_FORMALLY_VALID_CONTINUE_STAGE9_3_3`, with
`CONTINUE_STAGE9_3_3=YES`; official solver invocation remains zero and all 45
diagnostic calls are isolated. No official artifact, Stage 10 manifest, manual
acceptance, or Git index was changed.

## Stage 9.3.3 shadow equivalence and long-finger attribution (2026-07-23)

Added the isolated `shadow_equivalence.py` workflow, CLI commands, versioned
repeat-derived numerical contract, context binding, profile-isolation audit,
atomic checkpoint/resume layout, per-finger/counterfactual/gradient/constraint
reports, and HTML output boundary. The official baseline was rerun from three
selected frames with three repeats. QuerySet IDs/order and strict feasibility
flags matched, but qpos and canonical SDF differences were far above the
predeclared caps. The Stage 10 manifest also records inconsistent commits
(`23e6465` versus runtime environment `58fa77c`). Result:
`SHADOW_BASELINE_NOT_NUMERICALLY_EQUIVALENT`,
`RETURN_TO_STAGE9_3_2_SHADOW_HARNESS_FIX`, `ENTER_STAGE9_4=NO`; mandatory
shadow profiles ran zero times. Formal artifacts and Git index remained
unchanged; code remains unstaged.

## Stage 9.3.4 provenance-rebased causal experiments (2026-07-23)

Added the isolated provenance audit, detached historical-lane contract,
current-lineage baseline runner, bounded multistart/base-seed diagnostics,
mandatory QuerySet/margin profiles, conservative readiness report, and HTML
handoff. Historical replay is fail-closed when its exact environment is not
available. Formal artifacts, Stage 10 manifest, manual acceptance, and Git
index remain unchanged; all new outputs are diagnostic and unstaged.

The current-lineage baseline passed all 60 frames: status 0, strict
acceptance, contiguous checkpoint chain, finite full-512 audit, and zero raw
penetration. Historical exact replay remained unavailable because the recorded
package environment did not match. Five selected frames completed 60
multistart variants, 60 base-seed final variants, and 15 formal mandatory
profile variants; projection rows remain explicitly unsolved diagnostics. The
final route is `STAGE9_3_4_INCONCLUSIVE` with `ENTER_STAGE9_4=NO` and a human
decision gate. No accepted artifact or Git index was changed.

## Stage 9.3.5 projection feasibility and causal closure (2026-07-23)

Added the isolated Stage 9.3.5 feasibility path scan, canonical full-512
state projections, counterfactual states, objective and constraint
attribution, gated branch-rollout contract, reports, HTML handoff, tests, and
synchronized documentation. The workflow is diagnostic-only and preserves
the formal Stage 9.2 solver contract, Stage 10 artifacts, and Git index.

The bounded run completed the frame-10 1001-sample canonical full-512 path
scan and read-only attribution bundle: 12 counterfactual states, 6 objective
rows, and 512 constraint-pressure rows. The projection attempt was paused
before a valid solver checkpoint because the CPU full-512 constraint evaluation
did not complete within the bounded wall-time; no projection result is treated
as accepted. Final route:
`RETURN_TO_PROJECTION_DIAGNOSTIC_HARNESS_FIX`, `ENTER_STAGE9_4=NO`.

## Stage 9.3.5 continuation and five-frame closure (2026-07-23)

The diagnostic harness was repaired without changing the canonical evaluator:
resume now validates and reuses full-512 path caches, and projection callbacks
reuse the already cross-validated `convex_hull_exact_solver_only` solver-side
backend while every candidate is independently checked with the canonical
`reference_triangle_winding` backend. The minimal profile also received its
zero-slack Jacobian padding fix. All changes remain diagnostic-only and
unstaged.

The requested selected frames `[10,39,30,36,0]` now each have a 1001-sample,
512-sample path cache and compact feasibility report. Both projection profiles
ran for all five frames (10 attempts total); 3 attempts are strict accepted
projections and the remaining attempts are recorded as `status=9` solver
failures, never as accepted solutions. Independent canonical validation passed
finite/full-512, bounds, hard/soft, slack, and raw-penetration checks for the
reported candidate states.

The complete read-only causal bundle contains 90 counterfactual states, 90
objective attribution rows, and 2,560 constraint-pressure rows. The bounded
branch gate correctly returned `NOT_REQUIRED_BY_GATE` because no candidate met
the improvement gate. Final route is
`READY_FOR_STAGE9_4_REFINEMENT_ENGINEERING_REPAIR`, with
`ENTER_STAGE9_4=NO`, `HUMAN_DECISION_REQUIRED=YES`, and
`STOP_AFTER_STAGE9_3_5=TRUE`. Formal Stage 7/8/9/10 artifacts, current-lineage
baseline, manual acceptance, and the Git index remain unchanged.

## Stage 9.3.5 causal-gate correction (2026-07-23)

The final-report audit found that the initial aggregate route above was too
permissive: a low state fraction alone cannot establish
`OFFICIAL_FINAL_MOVES_BEYOND_FEASIBILITY`. The declared gate also requires at
least two representative frames and long-finger RMSE improvement of at least
`max(1.0 mm, 10%)`, in addition to strict canonical feasibility and a
projection closer to warm. The refreshed checkpoint validations show 3 strict
accepted attempts on 2 distinct frames, but every reported long-finger
improvement is negative; therefore zero frames pass the complete causal gate.

The assembled reports now classify frames 39, 30, 36, and 0 as
`WARM_ALREADY_FEASIBLE`, frame 10 as `INCONCLUSIVE`, and the aggregate route as
`STAGE9_4_NOT_YET_JUSTIFIED`. `ENTER_STAGE9_4=NO`,
`HUMAN_DECISION_REQUIRED=YES`, and `STOP_AFTER_STAGE9_3_5=TRUE`. No official
artifact, current-lineage baseline, Stage 10 artifact, manual acceptance, or
Git index was changed.

## 2026-07-24 -- Stage 9 one-shot causal closure

Completed the bounded projection contract, Eq. (9) implementation map, fixed
C0--C7 selected-frame sweep, single-root-cause decision, and the one allowed
faithful regularization repair. Projection remains diagnostic-only; full 60
frame validation and human review status are recorded in
`.local/reports/stage9_one_shot/`.

## 2026-07-24 -- G3 derived sign proxy and strict conflict audit

Added the deterministic derived watertight sign-proxy workflow and hybrid
original-distance/proxy-sign backend. The banana source mesh remained read-only;
Candidate 1 local repair passed the fixed 20k surface gates and produced a
watertight proxy with recorded boundary loops, near-zero IDs, patch IDs, hashes,
and provenance. Identity validation passed for G1/G2/G4.

The resumed A–E lane reached G3 Stage A and then correctly failed closed as
`SIGN_PROXY_CONTACT_REGION_CONFLICT`: three active QuerySet samples were within
the original boundary exclusion zone and two used synthetic patch faces. No
trajectory, margin, frame, raw asset, or historical Stage 10 artifact was
modified. The geometry audit HTML and first/middle/last PNGs were generated;
G3 downstream C–E and the aggregate A–E recommendation remain invalid.

## 2026-07-28 -- Wuji Hand2 three-clip generic suite

Added the frozen W1/W2/W3 suite configuration and generic `run-grab-suite`
orchestrator. It binds source/MANO/object/robot/profile hashes before Stage 9,
supports checkpoint resume, independent full formal-surface validation,
reference export, metrics, and self-contained HTML. The implementation keeps
the `pene-loss` worktree and all historical artifacts outside its write scope.

## 2026-07-29 -- W2.3 sequential finalization

Added `wuji_continuous_sequential_v1`, with window fallback disabled in the
production sequential path and a separate fixed-anchor five-frame diagnostic
shadow. The bounded runner writes replay, penetration, oracle, export, HTML,
and integrity evidence only to the W2.3 output root; no formal artifact is
replaced and no Git mutation is performed.

## 2026-07-30 -- Stage-12 final-job quiescence and performance repair

Installed fail-closed final-job control, stopped only identified legacy Stage-12 process groups,
and retained them as `SIGSTOP` because no safe per-frame checkpoint was present. The queue remains
paused. The new CPU policy is one worker / one BLAS/Torch thread; the fast-exact execution profile
is a non-default candidate pending real-frame parity and controlled scheduler evidence.

## 2026-07-30 -- P2 analytic SDF and exact BVH qualification

Added the v2 spatial-gradient chain rule, ambiguity-only 3D FD, certified
object-local sign cache, and exact object-local BVH instrumentation. The fixed
five-frame diagnostic remains separate from Stage-12 and all frames are recorded
under `.local/experiments/final_refinement_perf_v2/`.
