# TopoRetarget-Repro

[中文 README](README.zh-CN.md)

TopoRetarget-Repro is an unofficial, independent, paper-traceable reproduction repository for
[*TopoRetarget: Interaction-Preserving Retargeting for Dexterous Manipulation*](https://arxiv.org/abs/2606.16272).
It provides a robot-independent HOI data contract, explicit coordinate conventions, source-hand
conversion tools, reproducibility audits, and a staged path toward full dexterous retargeting.

The repository is intentionally transparent about scope: the current implementation reaches the
canonical HOI interface, a bounded MANO-to-MediaPipe-style-21 source adapter, the Stage 4 generic
robot-hand/Arti-MANO target kinematics interface, the bounded Stage 5 GRAB dataset adapter, a
Stage 7 relative-bone-direction warm-start trajectory, the bounded Stage 8 source interaction
graph/Laplacian loss, the bounded Stage 9 Eq. (8)-(9) final refinement, and the W0/W1 generic Wuji
Hand2 Beta1 target-hand integration. Stages 7-9 remain bounded implementations with explicit
paper assumptions; W0/W1 does not claim full Wuji retargeting, hardware reproduction, RL, or the
paper's reported experimental result.

## Overview

The main entry point is the `toporetarget` CLI. The code is organized around complete capabilities:

- canonical, robot-independent `HOISequence` data with scene-frame geometry and explicit SE(3)
  frame conversions;
- read-only inspection of one GRAB NPZ sequence and conversion to a canonical Zarr cache;
- explicit MANO semantic layouts and versioned MANO-to-MediaPipe21 mapping profiles;
- generic differentiable URDF hand FK, named qpos, target anchors, and Arti-MANO/Wuji RH/LH inspection;
- a lazy GRAB index, native-time/native-mesh single-sequence adapter, contact modes, validation,
  provenance, and raw/canonical comparison;
- source/object/timestamp preservation reports and static or interactive geometry viewers;
- relative bone-direction Eq. (1), sequential Eq. (2) qpos warm starts, base observability reports,
  canonical-frame alignment, and independent `toporetarget.warm_start.v1` artifacts;
- source-only Eq. (3)-(7) interaction graphs with fixed 21+50 vertices, source-derived directed
  weights, shared Laplacians, frozen warm-start evaluation, qpos Jacobians, and RH/LH bounded reports;
- paper-fidelity auditing, assumptions tracking, and tracked Arti-MANO asset provenance support.

Arti-MANO and Wuji Hand2 Beta1 are tracked robot-hand assets under `third_party/robot_hands/`; the
Wuji bundle is limited to the approved Hand2 Beta1 body subset. External datasets, MANO/SMPL-X
models, and extraction caches are not distributed here. Keep machine-local data under `.local/`
configured paths. The canonical data
interface is described in [`docs/HOI_DATA_INTERFACE.md`](docs/HOI_DATA_INTERFACE.md), and frame
semantics are defined in [`docs/COORDINATE_CONVENTIONS.md`](docs/COORDINATE_CONVENTIONS.md).

The Wuji target boundary is documented in
[`docs/WUJI_HAND2_BETA1_TARGET.md`](docs/WUJI_HAND2_BETA1_TARGET.md), with Chinese documentation
in [`docs/WUJI_HAND2_BETA1_TARGET.zh-CN.md`](docs/WUJI_HAND2_BETA1_TARGET.zh-CN.md).

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
| 7 | Relative bone-direction initialization | Complete, with assumptions | 20-bone/15-pair Eq. 1, sequential bounded Eq. 2, frame audit, RH/LH acceptance, artifacts, validation, and visual diagnostics pass. |
| 8 | Interaction graph and Laplacian coordinates | Complete, bounded; assumptions explicit | Source-only Eq. 3–7 graph/loss, RH/LH artifacts, identity/Jacobian validation, reports, and views pass. Eq. 8–9 remains Stage 9. |
| 9 | Constrained optimization with slack variables | Complete, bounded; assumptions explicit | Eq. 8–9 final refinement, full/adaptive collision QuerySet, slack, independent full-surface audit, RH/LH bounded trajectory artifacts, CLI, tests, and views pass; no Stage 10 behavior is included. |
| 10 | GRAB → Arti-MANO end-to-end retargeting | Implemented, bounded reference-runtime accepted; preferred performance open | Resumable bounded DAG, official contact-window selection, provenance, review/export, and the accepted `s1/airplane_lift` right-hand 60-frame reference-runtime milestone; preferred performance, production, and real-time scopes remain open. |
| Q1–Q3 | Multi-dataset interaction benchmark and unified automatic evaluation | Implemented, bounded; current local ContactPose selection gate is blocked before freeze | Frozen-selection contract, metric registry, automatic gates, manifest-bound profiles, reports, and HTML dashboard. The current local audit found no recognized official ContactPose contact attribution, so no baseline was run. This is not the paper's full 25-grasp ContactPose result. |
| Q4 | Morphology-aware warm-start | Not started | Evaluate morphology-aware initialization without changing the frozen Q1–Q3 baseline. |
| Q5 | Arti-MANO surface contact proxies | Not started | Validate robot surface/pad proxies separately from source labels. |
| Q6 | Contact-aware final extension | Not started | Add a separately versioned contact-aware extension after Q4/Q5 evidence. |
| Q7 | Cross-trajectory automatic profile selection | Not started | Select a profile only from frozen cross-trajectory evidence. |
| 12 | OakInk, DexYCB, and HO-Cap adapters | TODO | Add independently validated dataset adapters. |
| 13 | ARCTIC, OakInk2, and TACO extensions | TODO | Add independently validated dataset adapters. |
| 14 | Arbitrary dexterous-hand plugin interface | TODO | Test URDF/MJCF hand plugin contracts. |
| 15 | Baselines and ablations | TODO | Add fair OmniRetarget, Mink, DexPilot, and GeoRT runs. |
| 16 | Reference-tracking PPO | TODO | Add RL training and evaluation pipeline. |
| 17 | Paper experiment reproduction | TODO | Reproduce tables, figures, seeds, and result reports. |
| 18 | Performance optimization and v1.0 release | TODO | Establish benchmarks, packaging, and release criteria. |
| 19 | Non-paper extensions | TODO | Keep MANO cleanup, SPIDER, and other extensions separately labeled. |

The maintained roadmap with deliverables and status is [docs/ROADMAP.md](docs/ROADMAP.md); the
Chinese roadmap is [docs/ROADMAP.zh-CN.md](docs/ROADMAP.zh-CN.md). The benchmark contract and
automatic-evaluation notes are [docs/MULTI_DATASET_INTERACTION_BENCHMARK.md](docs/MULTI_DATASET_INTERACTION_BENCHMARK.md),
[docs/UNIFIED_AUTOMATIC_EVALUATION.md](docs/UNIFIED_AUTOMATIC_EVALUATION.md), and
[docs/EQ9_TEMPORAL_SCOPE_INTERPRETATIONS.md](docs/EQ9_TEMPORAL_SCOPE_INTERPRETATIONS.md), with
Chinese counterparts alongside them.

### Q1–Q3 frozen benchmark

The benchmark is a bounded engineering evaluation, not a claim of full paper-result reproduction.
It keeps dynamic GRAB clips and static ContactPose grasps separate, freezes selection before any
profile run, and reports ContactPose exact formulas separately from GRAB contact proxies. Use the
machine-local paths from the task environment or `.local/config.yaml`:

```bash
export PYTHONNOUSERSITE=1 PYTHONPATH=src
export GRAB_ROOT=/mnt/nas/storage/Ref2Dex_storage/GRAB/data/GRAB
export CONTACTPOSE_ROOT=/mnt/nas/storage/Ref2Dex_storage/ContactPose/data
export MANO_MODEL_ROOT=/mnt/nas/storage/Ref2Dex_storage/shared_assets/body_models/mano
export TOPORETARGET_ARTIMANO_ASSET_ROOT=third_party/robot_hands/artimano

python -m toporetarget benchmark inspect-datasets \
  --grab-root "$GRAB_ROOT" --contactpose-root "$CONTACTPOSE_ROOT" \
  --output .local/benchmarks/hoi_benchmark_v1/dataset_audit.json
python -m toporetarget benchmark select --config configs/benchmarks/hoi_benchmark_v1.yaml
python -m toporetarget benchmark freeze
python -m toporetarget benchmark run --resume
python -m toporetarget benchmark evaluate --html
python -m toporetarget benchmark dashboard
```

The selection lock is immutable for a run: later solver results cannot choose or replace a unit.
On the current local snapshot, GRAB selection passed for the fixed clip plus three additional
clips, but ContactPose selection is `Q1_CONTACTPOSE_SELECTION_BLOCKED` because 110 indexed
candidate annotations did not expose recognized official attribution fields. Therefore no
selection manifest, baseline, or result-level metric table is claimed; see the generated report
under `.local/benchmarks/hoi_benchmark_v1/`. Q4–Q7 remain unstarted.

## Quickstart

### Requirements

- Python 3.10–3.13
- Git
- External data/models only when using the corresponding workflows
- A graphical backend for `--show` viewers; `MPLBACKEND=Agg` is suitable for headless smoke tests
- SciPy for the Stage 7/9 numerical solvers

Install the complete environment for the currently implemented data and visualization workflows:

```bash
python -m pip install -e ".[dev,cache,viz,grab,robot,geometry,retarget]"
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
export TOPORETARGET_ARTIMANO_ASSET_ROOT=...    # optional explicit asset override
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

Every `--interactive`/`--show` Matplotlib window in this repository installs the same responsive
font handler: titles, axis labels, ticks, legends, annotations, frame labels, and widget labels
scale with the window area. Static PNG/PDF output is unchanged.

### Generate and Validate Relative Bone-Direction Warm Starts

Stage 7 consumes a canonical MediaPipe-21 cache and writes a separate bounded
initialization artifact. It does not read Stage 6 samples or SDF values as an
optimization target.

```bash
GRAB_CACHE=.local/cache/hoi/grab/cubemedium_inspect_1_rh_f000000_f000060_mp21.zarr
WARM_START=.local/cache/retarget/warm_start/s7_cubemedium_inspect_1_right_artimano_rh.zarr

toporetarget retarget inspect-bones \
  --canonical "$GRAB_CACHE" --hand right --frame 0 \
  --frame-profile canonical_keypoint_wrist_v1 \
  --bone-profile mediapipe21_full_finger_chain_v1 \
  --json .local/reports/stage7/source_bone_features_right.json \
  --csv .local/reports/stage7/source_bone_features_right.csv

toporetarget retarget compare-frame-profiles \
  --canonical "$GRAB_CACHE" --hand right --robot artimano_rh --frame 0 \
  --report .local/reports/stage7/frame_profile_comparison.json

toporetarget retarget warm-start \
  --canonical "$GRAB_CACHE" --hand right --robot artimano_rh \
  --start-frame 0 --end-frame 60 \
  --frame-profile canonical_keypoint_wrist_v1 \
  --bone-profile mediapipe21_full_finger_chain_v1 \
  --solver-profile paper_repro_scipy_trf --output "$WARM_START"

toporetarget retarget validate-warm-start \
  --canonical "$GRAB_CACHE" --warm-start "$WARM_START" \
  --report .local/reports/stage7/artimano_rh_validation.json \
  --csv .local/reports/stage7/artimano_rh_validation.csv
```

For debugging, use `visualize-warm-start` with `--view scene` or
`--view local-hand`, `--show-directions`, `--show-residuals`, and
`--show-hand-frames`. See
[`docs/RELATIVE_BONE_DIRECTION_INITIALIZATION.md`](docs/RELATIVE_BONE_DIRECTION_INITIALIZATION.md)
and [`docs/WARM_START_OPTIMIZATION.md`](docs/WARM_START_OPTIMIZATION.md).

The interactive viewer reuses the same scene/local layers and accepts the same display flags:

```bash
toporetarget retarget visualize-warm-start \
  --canonical "$GRAB_CACHE" --warm-start "$WARM_START" \
  --view scene --start-frame 0 --end-frame 60 --interactive \
  --show-source-hand --show-robot-hand \
  --show-source-skeleton --show-robot-skeleton \
  --show-hand-frames --show-labels --show-residuals \
  --show-object-context
```

For the static first/middle/last-frame diagnostics, change `--frame` and `--output` as follows:

```bash
# Scene overlay: source/robot keypoints, skeleton, frames, residuals, object context.
toporetarget retarget visualize-warm-start \
  --canonical "$GRAB_CACHE" --warm-start "$WARM_START" \
  --view scene --frame 0 \
  --show-hand-frames --show-labels --show-residuals --show-object-context \
  --output .local/reports/stage7/scene_first.png

# Local wrist-centered overlay with bone directions and adjacent features.
toporetarget retarget visualize-warm-start \
  --canonical "$GRAB_CACHE" --warm-start "$WARM_START" \
  --view local-hand --frame 0 \
  --show-directions --show-adjacent-features --show-labels --show-residuals \
  --output .local/reports/stage7/local_first.png
```

Use `--frame 30` and `--frame 59` for middle/last frames. In the interactive window, all
keypoint/skeleton/frame/label/residual fonts resize with the window; `--show-object-context` is
display-only and does not enter the warm-start objective.

### Stage 7.1. Audit warm-start fidelity and reachability

The manifest-driven Stage 7.1 audit replays the accepted Stage 7 warm-start,
checks source/robot mapping, thumb URDF ancestry and axes, frame/base alignment,
joint limits, per-finger attribution, and bounded diagnostic-only reachability.
It keeps formal Stage 7 fidelity separate from Stage 8/contact/task fidelity and
does not modify official artifacts. See
[`docs/WARM_START_FIDELITY_AND_REACHABILITY_AUDIT.md`](docs/WARM_START_FIDELITY_AND_REACHABILITY_AUDIT.md)
for the full contract and current accepted-run result.

```bash
env PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/deepcybo/miniconda3/envs/topo-retarget/bin/python \
  -m toporetarget workflow audit-warm-start \
  --run .local/runs/stage10_reference_runtime/s1__airplane_lift__right__artimano_rh__f000240_f000300/manifest.json \
  --canonical-contact-audit .local/runs/stage9_3_2_canonical_reaudit/s1__airplane_lift__right__artimano_rh__f000240_f000300 \
  --output-root .local/runs/stage7_1_warmstart_audit/s1__airplane_lift__right__artimano_rh__f000240_f000300 \
  --html --run-reachability-diagnostics --diagnostic-frames auto
```

### Stage 8. Build and validate the shared interaction graph

Stage 8 consumes the Stage 7 warm start and Stage 6 50-point sample artifact as separate,
hash-checked inputs. It builds source-only graph artifacts, then evaluates the frozen Eq. (7)
loss on the robot with the exact same connectivity and directed weights:

```bash
toporetarget retarget audit-interaction-inputs \
  --right-canonical .local/cache/hoi/grab/cubemedium_inspect_1_rh_f000000_f000060_mp21.zarr \
  --left-canonical .local/cache/hoi/grab/s7/cubemedium_inspect_1/semantic_left_f000000_f000060.zarr \
  --right-warm-start .local/cache/retarget/warm_start/s7_cubemedium_inspect_1_right_artimano_rh.zarr \
  --left-warm-start .local/cache/retarget/warm_start/s7_cubemedium_inspect_1_left_artimano_lh.zarr \
  --object-samples .local/cache/geometry/object_surface/cubemedium_samples.npz \
  --report .local/reports/stage8/input_audit.json

toporetarget retarget build-interaction-graph \
  --canonical .local/cache/hoi/grab/cubemedium_inspect_1_rh_f000000_f000060_mp21.zarr \
  --hand right --object-samples .local/cache/geometry/object_surface/cubemedium_samples.npz \
  --output .local/cache/retarget/interaction_graph/s7_cubemedium_inspect_1_right.zarr \
  --report .local/reports/stage8/rh_graph_build.json

toporetarget retarget evaluate-interaction \
  --graph .local/cache/retarget/interaction_graph/s7_cubemedium_inspect_1_right.zarr \
  --warm-start .local/cache/retarget/warm_start/s7_cubemedium_inspect_1_right_artimano_rh.zarr \
  --robot artimano_rh \
  --output .local/cache/retarget/interaction_evaluation/s7_cubemedium_inspect_1_right_artimano_rh.zarr
```

Interactive graph inspection (use `--mode source` and omit `--evaluation` to inspect only the
source graph):

```bash
toporetarget retarget visualize-interaction \
  --graph .local/cache/retarget/interaction_graph/s7_cubemedium_inspect_1_right.zarr \
  --mode compare \
  --evaluation .local/cache/retarget/interaction_evaluation/s7_cubemedium_inspect_1_right_artimano_rh.zarr \
  --start-frame 0 --end-frame 60 --interactive \
  --show-laplacian --show-residuals --show-contributions \
  --report .local/reports/stage8/rh_interactive_viewer.json
```

See [`docs/INTERACTION_GRAPH.md`](docs/INTERACTION_GRAPH.md),
[`docs/LAPLACIAN_INTERACTION_LOSS.md`](docs/LAPLACIAN_INTERACTION_LOSS.md), and the
[`Stage 8 report`](docs/stages/STAGE_8_INTERACTION_GRAPH_LAPLACIAN.md).

### Stage 9. Generate and Validate Final Interaction-Preserving Robot References

Stage 9 consumes the frozen Stage 7 warm-start, Stage 8 graph, and Stage 6 collision-surface
artifacts. It evaluates Eq. (8)–(9) in explicit local seed-delta coordinates with SLSQP,
positive-outside SDF hard/soft constraints, per-sample slack variables, monotonic adaptive
QuerySets, and an independent full-surface audit. The solver-only convex-hull acceleration is
accepted only after probe comparison with the Stage 6 reference backend; acceptance remains
reference-backend based.

```bash
toporetarget retarget inspect-query-set \
  --canonical .local/cache/hoi/grab/cubemedium_inspect_1_rh_f000000_f000060_mp21.zarr \
  --warm-start .local/cache/retarget/warm_start/s7_cubemedium_inspect_1_right_artimano_rh.zarr \
  --robot artimano_rh --frame 0 --query-profile adaptive_active_set_v1 \
  --json .local/reports/stage9/rh_query_set_frame0.json

toporetarget retarget refine \
  --canonical .local/cache/hoi/grab/cubemedium_inspect_1_rh_f000000_f000060_mp21.zarr \
  --warm-start .local/cache/retarget/warm_start/s7_cubemedium_inspect_1_right_artimano_rh.zarr \
  --graph .local/cache/retarget/interaction_graph/s7_cubemedium_inspect_1_right.zarr \
  --robot artimano_rh --query-profile adaptive_active_set_v1 \
  --coordinate-profile local_seed_delta_v1 \
  --solver-profile scipy_slsqp_active_set_v1 --start-frame 0 --end-frame 60 \
  --output .local/cache/retarget/final/s7_cubemedium_inspect_1_right_artimano_rh.zarr

toporetarget retarget validate-refinement \
  --canonical .local/cache/hoi/grab/cubemedium_inspect_1_rh_f000000_f000060_mp21.zarr \
  --warm-start .local/cache/retarget/warm_start/s7_cubemedium_inspect_1_right_artimano_rh.zarr \
  --graph .local/cache/retarget/interaction_graph/s7_cubemedium_inspect_1_right.zarr \
  --final .local/cache/retarget/final/s7_cubemedium_inspect_1_right_artimano_rh.zarr \
  --report .local/reports/stage9/rh_validation.json \
  --csv .local/reports/stage9/rh_validation.csv

toporetarget retarget audit-penetration \
  --canonical .local/cache/hoi/grab/cubemedium_inspect_1_rh_f000000_f000060_mp21.zarr \
  --warm-start .local/cache/retarget/warm_start/s7_cubemedium_inspect_1_right_artimano_rh.zarr \
  --final .local/cache/retarget/final/s7_cubemedium_inspect_1_right_artimano_rh.zarr \
  --robot artimano_rh --report .local/reports/stage9/rh_full_surface_audit.json

toporetarget retarget visualize-refinement \
  --canonical .local/cache/hoi/grab/cubemedium_inspect_1_rh_f000000_f000060_mp21.zarr \
  --warm-start .local/cache/retarget/warm_start/s7_cubemedium_inspect_1_right_artimano_rh.zarr \
  --graph .local/cache/retarget/interaction_graph/s7_cubemedium_inspect_1_right.zarr \
  --final .local/cache/retarget/final/s7_cubemedium_inspect_1_right_artimano_rh.zarr \
  --robot artimano_rh --frame 0 --output .local/reports/stage9/scene_first.png

# Interactive bounded clip; omit --output because the window is live.
toporetarget retarget visualize-refinement \
  --canonical .local/cache/hoi/grab/cubemedium_inspect_1_rh_f000000_f000060_mp21.zarr \
  --warm-start .local/cache/retarget/warm_start/s7_cubemedium_inspect_1_right_artimano_rh.zarr \
  --graph .local/cache/retarget/interaction_graph/s7_cubemedium_inspect_1_right.zarr \
  --final .local/cache/retarget/final/s7_cubemedium_inspect_1_right_artimano_rh.zarr \
  --robot artimano_rh --start-frame 0 --end-frame 60 --interactive \
  --show-labels --show-frames --show-collision-samples --show-query-set \
  --show-penetrations --show-slack --report .local/reports/stage9/rh_interactive_viewer.json
```

The full reference and solver comparison commands are documented in
[`docs/stages/STAGE_9_FINAL_CONSTRAINED_REFINEMENT.md`](docs/stages/STAGE_9_FINAL_CONSTRAINED_REFINEMENT.md).

The bounded RH/LH closeout on frames `[0,60)` passed full-surface validation and full-artifact
determinism. Minimum full signed distance was `0.623582905 m` (RH) and `0.641271031 m` (LH),
with zero penetration; adaptive/full comparison used 16/512 queries at frames `0/29/59`.
Detailed metrics, hashes, Jacobian checks, solver comparisons, and visual reports are in the
ignored `.local/reports/stage9/` directory. This closes Stage 9 only; RL, physics, ContactPose,
and baseline reproduction remain TODO. Stage 10 orchestration is available through
[`toporetarget workflow`](docs/END_TO_END_GRAB_ARTIMANO.md). The earlier
`s7/cubemedium_inspect_1` contact-rich attempt stopped at the existing Stage 9 solver's
iteration-limit failure; the later `s1/airplane_lift` right-hand `[240,300)` run is the accepted
bounded reference-runtime milestone, limited to one offline 60-frame window.

#### Stage 9.1 solver-robustness closeout

The v1 profile remains unchanged. Contact-rich runs may return a feasible SLSQP
candidate with status `9` / `Iteration limit reached`; strict acceptance still
rejects it because optimizer convergence is a separate required field. The v2
profile continues adaptive active-set solves from `result.x`, remaps old slack
by query ID, initializes only new slack with the minimum bounded value, and
persists the continuation trace. It preserves Eq. (8), Eq. (9), paper weights,
base parameterization, q/slack bounds, signed-distance sign, and the full
512-point audit.

Use the explicit profile selector when resuming Stage 10:

```bash
toporetarget workflow run-grab \
  --sequence s1/airplane_lift --index .local/index/grab \
  --hand right --robot artimano_rh --start-frame 240 --end-frame 300 \
  --refinement-solver-profile scipy_slsqp_active_set_contact_rich_v2 \
  --run-root .local/runs/stage10
```

The fixed-grid benchmark, strict status fields, deterministic repeats, selected
uniform maxiter, and profile hashes are recorded in
`.local/reports/stage9_1/maxiter_benchmark.json`. Stage 10 signatures include
the selected profile ID/hash; changing it invalidates Stage 9 and downstream
nodes while Stage 5–8 inputs remain reusable. Solver/termination details remain
paper-undisclosed implementation assumptions.
The fixed benchmark currently selects uniform `maxiter=100` (35 records). v1 is
`6affff2fdb425a0402f643c291c0b8904d4dbec6c5b69a5006cf9829dcc220aa`; v2 is
`c42c21d894c54d07b1d30943b5a3338b13628bf0429ab203b5540cf934d09b7c`. The full
60-frame real artifact and deterministic repeat are explicit opt-in gates; they
are now backed by the Stage 9.2 contact-rich run and full fresh/resumed comparison.
Stage 9.2 meets the reference-runtime minimum gate, while its preferred
single-frame median/p95 target remains unmet. The bounded reference-runtime Stage 10 milestone is
accepted; preferred performance debt remains open.

#### Stage 9.2 performance and recoverable execution

Stage 9.2 adds exact-x callback reuse, persistent SDF/FK resources, exact
reference-SDF AABB acceleration, batched collision Jacobians, explicit solver
conditioning, scheduled independent 512-point audits, atomic frame
checkpoints, soft wall-time pause/resume, assembly, and fresh/resumed
comparison. It preserves the Stage 9 math and strict status-9 policy. Use
[`docs/REFINEMENT_PERFORMANCE.md`](docs/REFINEMENT_PERFORMANCE.md) and
[`docs/REFINEMENT_CHECKPOINT_AND_RESUME.md`](docs/REFINEMENT_CHECKPOINT_AND_RESUME.md)
for the profiling and recovery commands. The full 60-frame minimum runtime gate
and deterministic-repeat evidence pass. The v3 first/repeat runs measure
`10.766/38.711 s` and `10.773/39.052 s` median/p95, with `60/60` strict-accepted
status-0 frames in each run and exact persisted-array equality excluding
`solve_time_s`. The bounded reference-runtime Stage 10 milestone is accepted; the preferred
performance gate and explicit production/real-time scopes remain open.

### Stage 10. Run a bounded GRAB → Arti-MANO workflow

The accepted bounded reference-runtime milestone is now materialized at
`.local/runs/stage10_reference_runtime/s1__airplane_lift__right__artimano_rh__f000240_f000300/`.
It reuses the accepted Stage 5–9 artifacts, records `solver_invocation_count=0`,
and exports `exports/robot_reference.zarr` plus NPZ. The runtime decision is
deliberately limited to this single offline 60-frame milestone; production,
real-time, online-control, and full-dataset claims remain false.

```bash
toporetarget workflow run-grab \
  --sequence s1/airplane_lift --index .local/index/grab \
  --hand right --robot artimano_rh --auto-contact-window --window-length 60 \
  --mano-model-root /path/to/MANO --asset-root third_party/robot_hands/artimano \
  --run-root .local/runs/stage10 \
  --manual-acceptance .local/reports/stage9/manual_acceptance.json
```

Use `workflow status`, `workflow validate`, `workflow visualize`, and
`workflow export-reference` with the generated manifest. Resume and provenance rules are in
[`docs/WORKFLOW_RESUME_AND_PROVENANCE.md`](docs/WORKFLOW_RESUME_AND_PROVENANCE.md).

To inspect the actual source MANO mesh together with the warm-start and final Arti-MANO visual
meshes in a browser, write a self-contained HTML file:

```bash
toporetarget workflow visualize-mesh \
  --run .local/runs/stage10_reference_runtime/s1__airplane_lift__right__artimano_rh__f000240_f000300/manifest.json \
  --interactive
```

The default output is `review/trajectory_mesh.html`. Blue, orange, and green are source,
warm-start, and final meshes; the page also provides frame playback, orbit/zoom controls,
object context points, and per-frame refinement metrics. This is a visual inspection aid and
does not replace the numeric Stage 9/10 gates.

The same page also contains the frozen Stage 8 interaction graph and Laplacian diagnostics.
Select a mode in the page, or choose the initial mode at generation time:

```bash
toporetarget workflow visualize-mesh \
  --run .local/runs/stage10_reference_runtime/s1__airplane_lift__right__artimano_rh__f000240_f000300/manifest.json \
  --mode combined \
  --output .local/runs/stage10_reference_runtime/s1__airplane_lift__right__artimano_rh__f000240_f000300/review/trajectory_combined.html
```

Available modes are `mesh`, `full-graph`, `figure4-style`, `laplacian-diagnostic`, and
`combined`; every mode keeps the source/warm-start/final mesh layers visible. Figure-4-style defaults to hand-object edges only; edge threshold/top-k,
display weights, residual target/scope, scalar heat, vector arrows, labels, and
source/warm/final graph states are controlled in the sidebar. The graph and object samples are
read from the accepted Stage 8 artifacts; this viewer never rebuilds or modifies them.
The complete explanation of the extra graph points/lines, residual overlays, sidebar controls,
and review semantics is in [`docs/INTERACTION_MESH_VISUALIZATION.md`](docs/INTERACTION_MESH_VISUALIZATION.md).

#### Visualize the entire trajectory

The existing `f000000_f000060` inputs and final artifacts contain only the half-open range
`[0,60)`. `visualize-refinement` cannot recover omitted frames. To view the whole source sequence,
first create one full canonical artifact, then run Stages 7–9 without frame bounds. This RH flow
reuses the existing Stage 6 object and collision-surface artifacts read-only; replace `right`/
`artimano_rh` with `left`/`artimano_lh` for the LH flow.

```bash
export FULL_CANONICAL=.local/cache/hoi/grab/s7/cubemedium_inspect_1/both_full_mp21.zarr
export OBJECT_SAMPLES=.local/cache/geometry/object_surface/cubemedium_samples.npz
export RH_WARM_FULL=.local/cache/retarget/warm_start/s7_cubemedium_inspect_1_right_artimano_rh_full.zarr
export RH_GRAPH_FULL=.local/cache/retarget/interaction_graph/s7_cubemedium_inspect_1_right_full.zarr
export RH_EVAL_FULL=.local/cache/retarget/interaction_evaluation/s7_cubemedium_inspect_1_right_full.zarr
export RH_FINAL_FULL=.local/cache/retarget/final/s7_cubemedium_inspect_1_right_artimano_rh_full.zarr
export RH_SURFACE=.local/cache/geometry/robot_surface/artimano_rh_neutral.npz

# 1. Convert and validate all frames. Omitting both frame flags means the full sequence.
toporetarget data convert --dataset grab --index .local/index/grab \
  --sequence s7/cubemedium_inspect_1 --hands both \
  --include-table --contact-mode semantic --include-mediapipe21 \
  --mano-model-root "$MANO_MODEL_ROOT" --output "$FULL_CANONICAL" --force
toporetarget data validate --dataset grab --index .local/index/grab \
  --sequence s7/cubemedium_inspect_1 --canonical "$FULL_CANONICAL" \
  --mano-model-root "$MANO_MODEL_ROOT" \
  --report .local/reports/stage5/grab_full_validation.json
toporetarget geometry validate-samples --canonical "$FULL_CANONICAL" \
  --object-id primary --samples "$OBJECT_SAMPLES" \
  --report .local/reports/stage6/full_object_samples_validation.json

# 2. Stage 7 full warm start.
toporetarget retarget warm-start --canonical "$FULL_CANONICAL" \
  --hand right --robot artimano_rh \
  --frame-profile canonical_keypoint_wrist_v1 \
  --bone-profile mediapipe21_full_finger_chain_v1 \
  --solver-profile paper_repro_scipy_trf --output "$RH_WARM_FULL" --force

# 3. Stage 8 full graph and frozen Eq. (7) evaluation.
toporetarget retarget build-interaction-graph --canonical "$FULL_CANONICAL" \
  --hand right --object-samples "$OBJECT_SAMPLES" \
  --output "$RH_GRAPH_FULL" --report .local/reports/stage8/rh_full_graph_build.json --force
toporetarget retarget evaluate-interaction --graph "$RH_GRAPH_FULL" \
  --warm-start "$RH_WARM_FULL" --robot artimano_rh \
  --output "$RH_EVAL_FULL" --force

# 4. Stage 9 sequential full refinement, validation, and independent collision audit.
toporetarget retarget refine --canonical "$FULL_CANONICAL" \
  --warm-start "$RH_WARM_FULL" --graph "$RH_GRAPH_FULL" --robot artimano_rh \
  --query-profile adaptive_active_set_v1 --coordinate-profile local_seed_delta_v1 \
  --solver-profile scipy_slsqp_active_set_v1 --collision-samples "$RH_SURFACE" \
  --output "$RH_FINAL_FULL" --force
toporetarget retarget validate-refinement --canonical "$FULL_CANONICAL" \
  --warm-start "$RH_WARM_FULL" --graph "$RH_GRAPH_FULL" --final "$RH_FINAL_FULL" \
  --robot artimano_rh --collision-samples "$RH_SURFACE" \
  --report .local/reports/stage9/rh_full_validation.json
toporetarget retarget audit-penetration --canonical "$FULL_CANONICAL" \
  --warm-start "$RH_WARM_FULL" --final "$RH_FINAL_FULL" --robot artimano_rh \
  --collision-samples "$RH_SURFACE" \
  --report .local/reports/stage9/rh_full_surface_audit.json

# 5. The viewer defaults to every frame in the final artifact when bounds are omitted.
toporetarget retarget visualize-refinement --canonical "$FULL_CANONICAL" \
  --warm-start "$RH_WARM_FULL" --graph "$RH_GRAPH_FULL" --final "$RH_FINAL_FULL" \
  --robot artimano_rh --interactive --show-labels --show-frames \
  --show-collision-samples --show-query-set --show-penetrations --show-slack
```

The indexed `s7/cubemedium_inspect_1` sequence has 951 frames at 120 FPS; the bounded clip is only
frames 0–59. The 60-frame RH/LH bounded run took about 20 minutes per side on this workstation, so
the full Stage 9 solve is roughly 5 h 17 min per side (about 10 h 34 min for both), excluding
conversion and upstream stages. This is only a linear estimate; inspect per-frame diagnostics.
The viewer does not run the solver: it only reads the final artifact. If the hand and object are still far
apart after this process, inspect the canonical scene overlay and the final viewer's object,
collision-sample, query-set, penetration, and slack layers. Do not enlarge the viewer bounds or
alter object poses/samples to manufacture a collision; a large positive SDF and zero penetration
mean the current source trajectory simply contains no collision at that frame.

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

Interactive comparison window:

```bash
toporetarget data compare \
  --dataset synthetic --sequence demo \
  --canonical .local/cache/hoi/synthetic_demo.zarr \
  --layout side-by-side --start-frame 0 --end-frame 8 --show \
  --show-keypoints --show-mesh --show-scene-frame --show-object-frame
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

For an interactive canonical-scene check of the bounded clip:

```bash
toporetarget data visualize --dataset grab --index .local/index/grab \
  --sequence s7/cubemedium_inspect_1 \
  --canonical .local/cache/hoi/grab/s7/cubemedium_inspect_1/both_f000000_f000060.zarr \
  --mode canonical --reference-frame scene --start-frame 0 --end-frame 60 \
  --interactive --show-mediapipe21 --show-mesh --show-table --show-contacts --show-axes
```

### 6. Tracked Arti-MANO asset setup

Normal execution uses the tracked snapshot. Inspect resolution and provenance with:

```bash
toporetarget robots resolve-assets
toporetarget robots compare-assets \
  --reference-root .local/assets/artimano
```

To reproduce the tracked snapshot from the pinned ManipTrans checkout:

```bash
toporetarget assets vendor-artimano \
  --source-root "$MANIPTRANS_ROOT" \
  --destination third_party/robot_hands/artimano \
  --imported-at 2026-07-27T19:00:00+08:00
```

The legacy `import-artimano` command and `.local/assets/artimano` directory remain only for
compatibility and migration. The resolver emits a deprecation warning when it falls back to them.
See [`docs/TRACKED_ROBOT_HAND_ASSETS.md`](docs/TRACKED_ROBOT_HAND_ASSETS.md),
[`docs/THIRD_PARTY_ASSET_POLICY.md`](docs/THIRD_PARTY_ASSET_POLICY.md), and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

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
The registry list command does not parse local assets; inspect and validation resolve the asset root
from `--asset-root`, `TOPORETARGET_ARTIMANO_ASSET_ROOT`, the tracked snapshot, or the legacy
fallback. Use `toporetarget robots resolve-assets` to see the selected source.

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

To open the same neutral collision view locally, replace `--output ...png` with `--show`; its
window text also scales responsively.

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

# Static SDF slice and collision-surface diagnostics.
toporetarget geometry visualize-sdf \
  --shape sphere --slice-axis z --slice-value 0 \
  --output .local/reports/stage6/sdf_sphere_slice_z0.png
toporetarget geometry visualize-robot-surface \
  --robot artimano_rh --pose neutral \
  --profile engineering_collision_32_per_geometry \
  --samples .local/cache/geometry/robot_surface/artimano_rh_neutral.npz \
  --output .local/reports/stage6/artimano_rh_collision_surface.png \
  --show-sample-normals
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

## Stage 9.3 contact-retention audit

The audit is a manifest-driven, solver-free diagnostic over the accepted
Stage 9.2/Stage 10 artifacts. It compares source, warm-start, final, visual
robot geometry, collision geometry, QuerySet provenance, same-definition
objective terms, and a non-optimizing warm-to-final interpolation path.

```bash
conda run -n topo-retarget env PYTHONNOUSERSITE=1 \
  python -m toporetarget workflow audit-contact-retention \
  --run .local/runs/stage10_reference_runtime/<run>/manifest.json \
  --output-dir .local/runs/stage9_3_contact_audit/<run> \
  --surface-samples 8192 --thresholds-mm 1,2,3,5,8,10 --html --force
```

The output records input immutability, positive-outside signed-distance
conventions, proxy assumptions, per-frame/per-link CSVs, root-cause analysis,
and a self-contained HTML review. Source contact and semantic-anchor retention
are explicitly diagnostic proxies, not ground-truth contact labels. See
[`docs/CONTACT_RETENTION_AUDIT.md`](docs/CONTACT_RETENTION_AUDIT.md) and the
[中文说明](docs/CONTACT_RETENTION_AUDIT.zh-CN.md).

For signed-distance reconciliation and the fail-closed maximum-three-frame
shadow boundary, see
[`docs/CONTACT_METRIC_RECONCILIATION_AND_SHADOW_ABLATION.md`](docs/CONTACT_METRIC_RECONCILIATION_AND_SHADOW_ABLATION.md).
The current reference-runtime closeout blocks shadow execution because the
legacy Stage 9.3 convex-hull metric does not share the Stage 9.2 reference SDF
definition.

### Stage 9.3.2 canonical re-audit

The formal audit source is now the versioned
[`reference_winding_v1`](configs/audit/contact_distance/reference_winding_v1.yaml)
reference winding SDF. Solver-side SDF acceleration is a separate contract;
the legacy convex-hull report is diagnostic-only and superseded for formal
contact/penetration claims. The v2 audit keeps raw penetration, tau/hard/soft
residuals, visual approximation, collision distance, and contact-retention
proxy semantics separate. Open visual meshes allow unsigned coverage audits
but do not prove inflated/inset direction. Shadow profiles are diagnostic-only
and can run only after the canonical 60x512 gate; no Stage 9.4 implementation
or Stage 10 artifact is changed. See
[`docs/CANONICAL_CONTACT_DISTANCE_AND_REAUDIT.md`](docs/CANONICAL_CONTACT_DISTANCE_AND_REAUDIT.md).

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
- [Interaction graph](docs/INTERACTION_GRAPH.md) / [中文交互图](docs/INTERACTION_GRAPH.zh-CN.md)
- [Laplacian interaction loss](docs/LAPLACIAN_INTERACTION_LOSS.md) /
  [中文 Laplacian loss](docs/LAPLACIAN_INTERACTION_LOSS.zh-CN.md)
- [Stage 8 report](docs/stages/STAGE_8_INTERACTION_GRAPH_LAPLACIAN.md) /
  [中文 Stage 8 报告](docs/stages/STAGE_8_INTERACTION_GRAPH_LAPLACIAN.zh-CN.md)
- [Paper fidelity](docs/PAPER_FIDELITY.md)
- [Data and license policy](docs/LICENSE_AND_DATA_POLICY.md)
- [Development log](docs/DEVELOPMENT_LOG.md) / [中文开发日志](docs/DEVELOPMENT_LOG.zh-CN.md)
- [Contributing](CONTRIBUTING.md) / [third-party notices](THIRD_PARTY_NOTICES.md)

## License

The repository code and documentation are released under the GNU General Public License v3.0;
see [`LICENSE`](LICENSE). Tracked Arti-MANO retains the upstream license and notices under
`third_party/robot_hands/artimano/`; external GRAB, MANO/SMPL-X, ManipTrans source, and other datasets
remain subject to their own licenses. See
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

## Stage 9.3.3 shadow equivalence

The diagnostic Stage 9.3.3 boundary is documented in
[`docs/SHADOW_EQUIVALENCE_AND_LONG_FINGER_ABLATION.md`](docs/SHADOW_EQUIVALENCE_AND_LONG_FINGER_ABLATION.md).
It requires an official numerical-equivalence gate before running six bounded
shadow profiles and preserves the Eq. (1)-(9)/Stage 9.2/Stage 10 boundaries. The
current accepted-window replay is fail-closed with
`SHADOW_BASELINE_NOT_NUMERICALLY_EQUIVALENT`; no mandatory shadow profile or
Stage 9.4 implementation is authorized.

## Stage 9.3.4 provenance-rebased causal experiments

Stage 9.3.4 is an audit-only current-lineage experiment with a separate
historical lane, bounded multistart/base-seed diagnostics, and conservative
Stage 9.4 routing. See
[`docs/STAGE9_PROVENANCE_MULTISTART_AND_CAUSAL_ABLATION.md`](docs/STAGE9_PROVENANCE_MULTISTART_AND_CAUSAL_ABLATION.md).

## Stage 9.3.5 projection and causal closure

Stage 9.3.5 adds warm-to-final feasibility scans, diagnostic projections,
state counterfactuals, objective/constraint attribution, and a gated branch
rollout. It is diagnostic-only and preserves Eq. (1)-(9), formal weights,
official artifacts, and Stage 10. See the synchronized
[`docs/PROJECTION_FEASIBILITY_AND_CAUSAL_CLOSURE.md`](docs/PROJECTION_FEASIBILITY_AND_CAUSAL_CLOSURE.md)
for the exact commands and output roots.

## Stage 9 one-shot closure and repair

The completed fixed-profile causal sweep, Eq. (9) audit, single faithful
repair, full 60-frame validation, and versioned Stage 10 review bundle are
documented in [`docs/STAGE9_ONE_SHOT_CAUSAL_CLOSURE_AND_REPAIR.md`](docs/STAGE9_ONE_SHOT_CAUSAL_CLOSURE_AND_REPAIR.md).

## Faithful reproduction finalization

The accepted canonical faithful v3-fixed profile, legacy v2 classification,
quality-neutral human review, finalized versioned fixed Stage 10 export, and
A/B/C decision semantics are documented in
[`docs/FAITHFUL_REPRODUCTION_FINALIZATION.md`](docs/FAITHFUL_REPRODUCTION_FINALIZATION.md).

## GRAB Arti-MANO quality A–E

The frozen four-trajectory quality experiment is implemented behind
`toporetarget quality`. It uses G1–G4 from subject `s1` at native FPS, retains
both paper-core Eq. (9) profiles, and writes all new artifacts to
`.local/experiments/grab_artimano_quality_v1/`.

```bash
PYTHONNOUSERSITE=1 /home/deepcybo/miniconda3/envs/topo-retarget/bin/python \
  -m toporetarget quality run-a-to-e \
  --config configs/experiments/grab_artimano_quality_v1.yaml \
  --resume --max-wall-time 1800 --generate-html
```

Use `toporetarget quality status` for the machine-readable recommendation and
open `html/index.html` for the four self-contained viewers. GRAB contact values
are dataset proxies, ContactPose is deferred, and the result scope is only a
within-subject multi-object development benchmark. See
[`docs/GRAB_ARTIMANO_QUALITY_EXPERIMENT.md`](docs/GRAB_ARTIMANO_QUALITY_EXPERIMENT.md)
for the full A–E contract.

For open-object geometry, the quality lane uses the documented
[`hybrid_original_distance_proxy_sign_v1`](docs/HYBRID_SIGNED_DISTANCE_FOR_OPEN_OBJECTS.md)
contract: the raw mesh is immutable, the original mesh supplies distance
magnitude and closest points, and a derived watertight proxy supplies sign only.
The current banana run is formally routed to
`SIGN_PROXY_CONTACT_REGION_CONFLICT` after the strict active-QuerySet boundary
gate; it is not an A–E completion claim. See
[`docs/DERIVED_WATERTIGHT_SIGN_PROXY.md`](docs/DERIVED_WATERTIGHT_SIGN_PROXY.md).

## Wuji Hand2 three-clip retargeting

The fixed within-subject `s1` benchmark uses `airplane_lift`, `apple_eat_1`,
and `alarmclock_lift` with `wuji_hand2_beta1_rh`. Run the generic suite with
[`docs/WUJI_HAND2_GRAB_RETARGETING.md`](docs/WUJI_HAND2_GRAB_RETARGETING.md).
Results are written to `.local/experiments/wuji_hand2_grab3_v1/`, including
metrics, independent 672-sample collision validation, reference exports, and
HTML viewers. This is an offline reference runtime; GRAB contact values are
dataset proxies, and Wuji visual soft pads are not formal collision geometry.
## Wuji Hand2 continuous retargeting

The engineering profile `wuji_continuous_full_state_v1` runs the frozen W1/W2/W3 suite with previous-final correction transport, chart-consistent full-state temporal continuation, continuity acceptance, deterministic retry, and a bounded five-frame fallback. It does not modify the paper-core objective or use post-filtering. Run it with the command in [WUJI_CONTINUOUS_RETARGETING.md](docs/WUJI_CONTINUOUS_RETARGETING.md). Outputs are written to `.local/experiments/wuji_hand2_continuous_v1/`; the interactive review is under its `html/` directory. This is a three-clip `s1` engineering result, not a cross-subject or RL-ready claim.

The W2.2 bounded closeout is documented in
[`docs/stages/W2_2_WUJI_CONTINUITY_CLOSEOUT.md`](docs/stages/W2_2_WUJI_CONTINUITY_CLOSEOUT.md)
and is diagnostic-only under
`.local/experiments/wuji_hand2_continuous_v1/closeout_v1/`. Its current status
is `WUJI_CONTINUOUS_PROFILE_NOT_RECOMMENDED_WINDOW_FALLBACK_FAILED`: W2's 13
absolute q-step transitions are source/warm-driven with zero correction-driven
jumps, but the real W3 window fallback returns SLSQP status 4 and fails center
continuity; W3 penetration rate also regresses. No formal artifact is replaced.

W2.3 sequential finalization is a separately named candidate derived from the
continuous profile. It disables the production five-frame fallback, keeps the
window as a nonblocking diagnostic shadow, and writes all new evidence and
versioned exports under
`.local/experiments/wuji_hand2_continuous_v1/w2_3_finalization/`. See
[`docs/stages/W2_3_WUJI_SEQUENTIAL_FINALIZATION.md`](docs/stages/W2_3_WUJI_SEQUENTIAL_FINALIZATION.md).

The recommended profile is `wuji_continuous_sequential_v1` for offline
reference generation only. Resume it with:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/deepcybo/miniconda3/bin/python scripts/wuji_w2_3_finalization.py
```

The current sequential status is
`WUJI_CONTINUOUS_SEQUENTIAL_PROFILE_RECOMMENDED_WITH_SECONDARY_PENETRATION_WARNING`.
The window remains `WINDOW_FALLBACK_EXPERIMENTAL_UNRESOLVED_NONBLOCKING`; it
does not block the sequential gate. The multi-threshold audit evidence is
under `.local/experiments/wuji_hand2_continuous_v1/w2_3_finalization/penetration_audit/`.
The five new HTML pages are under the same output root's `html/` directory.
This lane is not RL-ready, not realtime-ready, not cross-subject validated,
and does not establish author-exact reproduction.
