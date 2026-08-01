# TopoRetarget-Repro

[中文 README](README.zh-CN.md)

TopoRetarget-Repro is an unofficial, independent, paper-traceable reproduction of
[*TopoRetarget: Interaction-Preserving Retargeting for Dexterous Manipulation*](https://arxiv.org/abs/2606.16272).
It turns GRAB hand-object motion into offline dexterous-hand references through a canonical HOI
contract, MANO semantic conversion, target-hand kinematics, geometry/SDF processing,
relative-bone warm starts, interaction graphs, constrained refinement, validation, and
manifest-bound export.

The repository supports tracked Arti-MANO and Wuji Hand2 Beta1 target assets. External datasets
and MANO/SMPL-X models are not redistributed. Implemented results are bounded engineering
reproductions with explicit provenance and assumptions; they are not claims of author-exact,
full-dataset, real-time, hardware-control, physics, or RL reproduction.

## Overview

The `toporetarget` CLI provides:

- a robot-independent `HOISequence` schema with native timestamps, scene-frame geometry, and
  explicit SE(3) conventions;
- lazy GRAB indexing, bounded native-frame conversion, semantic contact handling, and
  MANO-to-MediaPipe-style-21 conversion;
- generic YAML-registered URDF target hands, differentiable/reference FK, named qpos, semantic
  anchors, collision surfaces, and tracked Arti-MANO/Wuji assets;
- deterministic object-surface sampling, signed-distance queries, collision QuerySets, and
  independent full-surface audits;
- relative-bone warm starts, frozen source interaction graphs/Laplacians, and constrained
  interaction-preserving refinement;
- resumable, content-hashed workflows, immutable source/provenance records, automatic validation,
  human-review boundaries, and versioned robot-reference exports;
- self-contained browser HTML for source/warm/final meshes, interaction graphs, continuity,
  collision/contact diagnostics, metrics, and audit evidence.

Machine-local datasets, models, caches, reports, and runs belong under ignored `.local/` paths.
The central contracts are [HOI data](docs/HOI_DATA_INTERFACE.md),
[coordinate conventions](docs/COORDINATE_CONVENTIONS.md), and the
[robot-hand target contract](docs/ROBOT_HAND_TARGET_CONTRACT.md).

## Isaac Lab GPU Backend

MuJoCo is retained as the CPU correctness, deterministic-regression, contact
diagnostic, action-replay, and visualization backend. GPU-parallel platform and
future policy work move to an isolated Isaac Lab lane; MuJoCo evidence does not
authorize PhysX assets, an oracle, or PPO.

Stage 16-C.0 freezes Python 3.11.15, Isaac Sim 5.1.0, Isaac Lab `v2.3.2` at
`37ddf626871758333d6ed89cf64ad702aef127d0`, and Torch 2.7.0 cu128. Its current
status is `STAGE16C0_ISAACLAB_PLATFORM_BLOCKED` because no NVIDIA EULA
authorization is recorded. Static audit remains reproducible and runtime
qualification stops before Isaac import:

NVIDIA now labels Isaac Sim 5.1 unsupported; this exact 5.1/v2.3.2 pair is a
frozen reproduction target, not a claim of continuing vendor support.

```bash
bash scripts/bootstrap_stage16_isaaclab_env.sh --dry-run
conda run -n toporetarget-isaaclab python scripts/verify_stage16_isaaclab_platform.py --phase static
conda run -n toporetarget-isaaclab python scripts/verify_stage16_isaaclab_platform.py --phase full --steps 1000
```

The last command records `ISAACLAB_EULA_ACCEPTANCE_REQUIRED`; it does not
accept the agreement. Headless and viewer smoke commands are documented in
[the environment setup](docs/rl/ISAACLAB_ENVIRONMENT_SETUP.md). Wuji/HO-Cap
asset migration, a custom Isaac Lab task, a PhysX oracle, and PPO are not
started.

## Dataset support

| Dataset | Adapter | Source qualification | Strict final qualification | Notes |
|---|---|---:|---:|---|
| GRAB | Complete | Validated | Validated | Primary initial dynamic reference dataset |
| DexYCB | Complete | 2/2 | 2/2 | Native PCA45 and subject-shape routing |
| OakInk | Complete | 2/2 | 2/2 | Native hand vertices/joints and object transforms |
| HO-Cap | Complete | 2/2 | 2/2 | PCA45, subject shape and qxyzw object pose |
| ContactPose | Complete | 2/2 static | 2/2 strict | Static one-frame samples; official joints; paper contact benchmark not reproduced |
| ARCTIC | TODO | — | — | Stage 13 |
| OakInk2 | TODO | — | — | Stage 13 |
| TACO | TODO | — | — | Stage 13 |

## Robot-hand support

| Target | Kinematics | Retarget | Collision | Simulation/RL |
|---|---|---|---|---|
| Arti-MANO | Validated | Validated | Validated | Not RL-qualified |
| Wuji Hand2 Beta1 | Validated | Validated | Validated | Offline reference generation only |
| Generic URDF/MJCF | Import foundation | Manifest required | Profile required | Not automatically guaranteed |

## Environment setup

### Requirements and installation

- Linux, Git, and Python `>=3.10,<3.14`; Python 3.12 is the maintained local workflow.
- SciPy, PyTorch, Zarr, trimesh, SMPL-X, and browser/visualization dependencies for the complete
  pipeline.
- GRAB and MANO files for real-data runs. Respect their upstream licenses.

Create an isolated environment and install every implemented workflow extra:

```bash
conda create -n topo-retarget python=3.12 -y
conda activate topo-retarget
python -m pip install -U pip
python -m pip install -e ".[dev,cache,viz,grab,robot,geometry,retarget]"
```

### Configure local resources

Do not copy licensed data or model files into Git. Set paths directly or create the ignored
`.local/config.yaml` from [configs/paths.example.yaml](configs/paths.example.yaml):

```bash
export GRAB_ROOT=/path/to/GRAB
# GRAB_ROOT is the dataset root consumed by `toporetarget data index`.
export MANO_MODEL_ROOT=/path/to/body_models/mano
# MANO_MODEL_ROOT must contain MANO_LEFT.pkl and MANO_RIGHT.pkl.

export PYTHONNOUSERSITE=1
export PYTHONPATH=src
export TOPORETARGET_PYTHON="${CONDA_PREFIX}/bin/python"
```

Arti-MANO and Wuji assets are tracked under `third_party/robot_hands/`. Arti-MANO can be
overridden with `TOPORETARGET_ARTIMANO_ASSET_ROOT`; otherwise the tracked bundle is used.

### Verify the installation and assets

```bash
"$TOPORETARGET_PYTHON" -m toporetarget --help
"$TOPORETARGET_PYTHON" -m toporetarget doctor paper
"$TOPORETARGET_PYTHON" -m toporetarget robots list
"$TOPORETARGET_PYTHON" -m toporetarget robots validate artimano_rh \
  --asset-root third_party/robot_hands/artimano
"$TOPORETARGET_PYTHON" -m toporetarget robots validate wuji_hand2_beta1_rh \
  --asset-root third_party/robot_hands/wuji_hand2_beta1
```

## Complete workflow

The main workflow below is command-oriented and contains no historical stage log. Raw GRAB and
MANO inputs are read-only. Generated artifacts remain under `.local/`.

All required visualization is browser-based, self-contained HTML. The standard is a full-canvas
scene with a right-side control panel, frame slider/playback, orbit/zoom, source/warm/final
layers, graph/contact/collision filters, metrics, and provenance—matching the interaction style
of
`.local/experiments/wuji_hand2_continuous_v1/html/W1_airplane_lift_continuity_comparison.html`.
PNG/GIF and temporary Matplotlib windows are not part of this README workflow.

### 1. Select the source clip and target hand

`SEQUENCE`, `START_FRAME`, and `END_FRAME` select the GRAB object/action and native half-open
window. The examples below use right-hand 60-frame windows; choose another indexed sequence/range
without resampling or result-based reselection.

```bash
# Source object/action: change these three values together.
export SEQUENCE=s1/airplane_lift
export START_FRAME=240
export END_FRAME=300
export HAND=right

# Other fixed examples:
# apple:      SEQUENCE=s1/apple_eat_1      START_FRAME=212  END_FRAME=272
# alarmclock: SEQUENCE=s1/alarmclock_lift  START_FRAME=407  END_FRAME=467

# Target family: artimano or wuji.
export TARGET_FAMILY=artimano

case "${TARGET_FAMILY}:${HAND}" in
  artimano:right)
    export ROBOT=artimano_rh
    export TARGET_ASSET_ROOT=third_party/robot_hands/artimano
    ;;
  artimano:left)
    export ROBOT=artimano_lh
    export TARGET_ASSET_ROOT=third_party/robot_hands/artimano
    ;;
  wuji:right)
    export ROBOT=wuji_hand2_beta1_rh
    export TARGET_ASSET_ROOT=third_party/robot_hands/wuji_hand2_beta1
    ;;
  wuji:left)
    export ROBOT=wuji_hand2_beta1_lh
    export TARGET_ASSET_ROOT=third_party/robot_hands/wuji_hand2_beta1
    ;;
  *)
    echo "unsupported TARGET_FAMILY/HAND pair" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

export GRAB_INDEX=.local/index/grab
export SOLVER_PROFILE=scipy_slsqp_active_set_contact_rich_v3_fixed
```

The data/index and target-validation commands are shared by Arti-MANO and Wuji:

```bash
"$TOPORETARGET_PYTHON" -m toporetarget data index \
  --dataset grab --grab-root "$GRAB_ROOT" --output "$GRAB_INDEX"

"$TOPORETARGET_PYTHON" -m toporetarget data validate \
  --dataset grab --index "$GRAB_INDEX" --sequence "$SEQUENCE" \
  --hands "$HAND" --mano-model-root "$MANO_MODEL_ROOT" \
  --contact-mode semantic --start-frame "$START_FRAME" --end-frame "$END_FRAME" \
  --report .local/reports/preflight/source_validation.json

"$TOPORETARGET_PYTHON" -m toporetarget robots validate "$ROBOT" \
  --asset-root "$TARGET_ASSET_ROOT" \
  --report .local/reports/preflight/"${ROBOT}"_validation.json
```

### 2A. Run the complete Arti-MANO pipeline

The manifest-driven Arti-MANO runner resolves and validates the source, converts canonical
HOI/MANO semantics, audits object geometry, samples the object and robot collision surfaces,
generates and validates the warm start, builds/evaluates the frozen interaction graph, performs
final refinement, runs independent collision/semantic checks, and builds the review bundle.
Select `TARGET_FAMILY=artimano` in section 1 before running this lane.

Planning is solver-free:

```bash
test "$TARGET_FAMILY" = artimano
export RUN_ROOT=.local/runs/artimano
export WINDOW_LENGTH="$((END_FRAME - START_FRAME))"

"$TOPORETARGET_PYTHON" -m toporetarget workflow plan-grab \
  --sequence "$SEQUENCE" --index "$GRAB_INDEX" \
  --hand "$HAND" --robot "$ROBOT" \
  --start-frame "$START_FRAME" --end-frame "$END_FRAME" \
  --window-length "$WINDOW_LENGTH" \
  --refinement-solver-profile "$SOLVER_PROFILE" \
  --mano-model-root "$MANO_MODEL_ROOT" --run-root "$RUN_ROOT" \
  --output .local/reports/preflight/workflow_plan.json --dry-run
```

Run or resume the complete DAG:

```bash
"$TOPORETARGET_PYTHON" -m toporetarget workflow run-grab \
  --sequence "$SEQUENCE" --index "$GRAB_INDEX" \
  --hand "$HAND" --robot "$ROBOT" \
  --start-frame "$START_FRAME" --end-frame "$END_FRAME" \
  --window-length "$WINDOW_LENGTH" \
  --refinement-solver-profile "$SOLVER_PROFILE" \
  --mano-model-root "$MANO_MODEL_ROOT" --asset-root "$TARGET_ASSET_ROOT" \
  --run-root "$RUN_ROOT" --resume

RUN_ID="${SEQUENCE//\//__}__${HAND}__${ROBOT}__f$(printf '%06d' "$START_FRAME")_f$(printf '%06d' "$END_FRAME")"
export RUN_DIR="$RUN_ROOT/$RUN_ID"
export RUN_MANIFEST="$RUN_DIR/manifest.json"
```

`workflow run-grab` currently validates Arti-MANO target names by design. Use the suite runner
below for Wuji; do not substitute a Wuji robot name into this command.

### 2B. Run the complete Wuji pipeline

The generic suite runner executes the same canonical conversion, geometry, warm-start,
interaction-graph, refinement, validation, export, and HTML evaluation components for the
frozen right-hand Wuji clips. Select one object with `--unit`, or omit it to run W1/W2/W3.

```bash
export TARGET_FAMILY=wuji
export HAND=right
export ROBOT=wuji_hand2_beta1_rh
export TARGET_ASSET_ROOT=third_party/robot_hands/wuji_hand2_beta1
export WUJI_EXPERIMENT_ROOT=.local/experiments/wuji_hand2_grab3_v1

"$TOPORETARGET_PYTHON" -m toporetarget workflow run-grab-suite \
  --suite configs/experiments/wuji_hand2_grab3_v1.yaml \
  --grab-root "$GRAB_ROOT" --index "$GRAB_INDEX" \
  --mano-model-root "$MANO_MODEL_ROOT" \
  --robot "$ROBOT" --solver-profile "$SOLVER_PROFILE" \
  --experiment-root "$WUJI_EXPERIMENT_ROOT" \
  --resume --max-wall-time 1800 \
  --evaluate --export-reference --generate-html

# To run only the selected airplane example, append:
# --unit W1_s1__airplane_lift__right__wuji_hand2_beta1_rh__f000240_f000300
```

The Wuji suite writes its authoritative machine status to
`$WUJI_EXPERIMENT_ROOT/reports/final_status.json`, exports under `exports/`, and the HTML entry
point under `html/index.html`.

### 3. Validate, audit, and export an Arti-MANO run

The following checks are manifest-driven and do not alter raw data. JSON/CSV remains
authoritative; HTML is the visual audit surface.

```bash
"$TOPORETARGET_PYTHON" -m toporetarget workflow status --run "$RUN_MANIFEST"

"$TOPORETARGET_PYTHON" -m toporetarget workflow validate \
  --run "$RUN_MANIFEST" \
  --report "$RUN_DIR/reports/end_to_end_validation.json" \
  --csv "$RUN_DIR/reports/end_to_end_validation.csv"

"$TOPORETARGET_PYTHON" -m toporetarget workflow audit-contact-retention \
  --run "$RUN_MANIFEST" \
  --output-dir "$RUN_DIR/audits/contact_retention" \
  --surface-samples 8192 --thresholds-mm 1,2,3,5,8,10 \
  --html --force

"$TOPORETARGET_PYTHON" -m toporetarget workflow export-reference \
  --run "$RUN_MANIFEST" --format zarr \
  --output "$RUN_DIR/exports/robot_reference.zarr"
```

The exported `toporetarget.robot_reference.v1` is an offline trajectory artifact, not a robot
command stream.

### 4. Generate the unified intermediate/final HTML review

One combined page covers the necessary intermediate and final views: source MANO mesh,
warm-start target mesh, final target mesh, object context, frozen interaction graph,
Figure-4-style hand-object edges, Laplacian residuals, per-frame refinement metrics, and
provenance.

```bash
"$TOPORETARGET_PYTHON" -m toporetarget workflow visualize-mesh \
  --run "$RUN_MANIFEST" --mode combined \
  --output "$RUN_DIR/review/trajectory_comparison.html" \
  --interactive
```

The contact audit adds a second self-contained page at
`$RUN_DIR/audits/contact_retention/trajectory_contact_audit.html`, with source/warm/final,
visual/collision surfaces, QuerySet, semantic anchors, thresholds, frames, and link/region
controls. Visual plausibility never replaces the numerical validation and collision reports.

For Wuji, `--generate-html` creates the corresponding full-canvas pages under
`$WUJI_EXPERIMENT_ROOT/html/`. Open `index.html`; the per-clip pages provide source/warm/final
layers and the same browser-oriented playback/control pattern.

### 5. Complete the human-review boundary

Machine validation cannot fabricate human acceptance. Generate a template, inspect the combined
and contact-audit HTML across the required and worst frames, then have a named human reviewer fill
the copied record:

```bash
"$TOPORETARGET_PYTHON" -m toporetarget workflow review-template \
  --run "$RUN_MANIFEST" --output "$RUN_DIR/review/manual_acceptance.template.json"

cp "$RUN_DIR/review/manual_acceptance.template.json" \
  "$RUN_DIR/review/manual_acceptance.json"
# A human reviewer now fills manual_acceptance.json; do not auto-write a pass.
```

Resume the same `workflow run-grab` command from section 2A with:

```text
--manual-acceptance "$RUN_DIR/review/manual_acceptance.json"
```

Content hashes, selected frame range, source identity, robot/profile identity, solver status,
collision/continuity gates, and manual-review lineage must all remain valid. A failed gate stays
failed; do not skip frames, replace objects, or reselect from results.

### 6. Reproduce the implemented evaluation lanes

These are separate frozen evaluations, not hidden steps in the single-run workflow.

Arti-MANO four-clip quality/morphology/contact evaluation:

```bash
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
"$TOPORETARGET_PYTHON" -m toporetarget quality run-a-to-e \
  --config configs/experiments/grab_artimano_quality_v1.yaml \
  --resume --max-wall-time 1800 --generate-html

"$TOPORETARGET_PYTHON" -m toporetarget quality status \
  --experiment-root .local/experiments/grab_artimano_quality_v1
# HTML: .local/experiments/grab_artimano_quality_v1/html/index.html
```

Wuji continuity evaluation and comparison HTML:

```bash
"$TOPORETARGET_PYTHON" -m toporetarget workflow run-grab-suite \
  --suite configs/experiments/wuji_hand2_grab3_v1.yaml \
  --grab-root "$GRAB_ROOT" --index "$GRAB_INDEX" \
  --mano-model-root "$MANO_MODEL_ROOT" \
  --robot wuji_hand2_beta1_rh \
  --solver-profile wuji_continuous_full_state_v1 \
  --experiment-root .local/experiments/wuji_hand2_continuous_v1 \
  --resume --max-wall-time 1800 \
  --evaluate --export-reference --generate-html
# HTML: .local/experiments/wuji_hand2_continuous_v1/html/index.html
```

Frozen GRAB/ContactPose benchmark and unified dashboard:

```bash
export CONTACTPOSE_ROOT=/path/to/ContactPose

"$TOPORETARGET_PYTHON" -m toporetarget benchmark inspect-datasets \
  --grab-root "$GRAB_ROOT" --contactpose-root "$CONTACTPOSE_ROOT" \
  --output .local/benchmarks/hoi_benchmark_v1/dataset_audit.json
"$TOPORETARGET_PYTHON" -m toporetarget benchmark select \
  --config configs/benchmarks/hoi_benchmark_v1.yaml
"$TOPORETARGET_PYTHON" -m toporetarget benchmark freeze
"$TOPORETARGET_PYTHON" -m toporetarget benchmark run --resume
"$TOPORETARGET_PYTHON" -m toporetarget benchmark evaluate --html
"$TOPORETARGET_PYTHON" -m toporetarget benchmark dashboard
```

Selection and attribution gates fail closed. Static ContactPose units and dynamic GRAB units
remain separate, and GRAB contact proxies are never relabeled as ContactPose ground truth.

### 7. Repository-level audit

```bash
"$TOPORETARGET_PYTHON" scripts/check_paper_fidelity.py
"$TOPORETARGET_PYTHON" -m pytest -m "not licensed_data"
"$TOPORETARGET_PYTHON" -m ruff check .
"$TOPORETARGET_PYTHON" -m ruff format --check .
"$TOPORETARGET_PYTHON" -m mypy src
git diff --check
```

Licensed-data tests are opt-in and require the configured local GRAB/MANO resources.

## Documentation map

- Project planning and history:
  [roadmap](docs/ROADMAP.md) /
  [中文路线图](docs/ROADMAP.zh-CN.md),
  [development log](docs/DEVELOPMENT_LOG.md) /
  [中文开发日志](docs/DEVELOPMENT_LOG.zh-CN.md),
  [reproduction log](docs/REPRODUCTION_LOG.md)
- Paper and contracts:
  [paper fidelity](docs/PAPER_FIDELITY.md),
  [implementation specification](docs/PAPER_IMPLEMENTATION_SPEC.md),
  [assumptions](docs/ASSUMPTIONS.md),
  [open author questions](docs/OPEN_QUESTIONS_FOR_AUTHORS.md)
- Data and geometry:
  [data layout](docs/DATA_LAYOUT.md),
  [GRAB adapter](docs/GRAB_DATASET_ADAPTER.md),
  [MANO conversion](docs/MANO_TO_MEDIAPIPE21.md),
  [object geometry/sampling](docs/OBJECT_GEOMETRY_AND_SAMPLING.md),
  [signed distance](docs/SIGNED_DISTANCE_AND_COLLISION_QUERIES.md)
- Retargeting:
  [relative-bone initialization](docs/RELATIVE_BONE_DIRECTION_INITIALIZATION.md),
  [interaction graph](docs/INTERACTION_GRAPH.md),
  [Laplacian loss](docs/LAPLACIAN_INTERACTION_LOSS.md),
  [final refinement](docs/CONTACT_PRESERVING_FINAL_REFINEMENT.md),
  [workflow resume/provenance](docs/WORKFLOW_RESUME_AND_PROVENANCE.md)
- HTML review and audits:
  [trajectory visualization](docs/TRAJECTORY_VISUALIZATION.md),
  [interaction-mesh HTML](docs/INTERACTION_MESH_VISUALIZATION.md),
  [contact-retention audit](docs/CONTACT_RETENTION_AUDIT.md),
  [warm-start audit](docs/WARM_START_FIDELITY_AND_REACHABILITY_AUDIT.md)
- Target hands and evaluations:
  [tracked robot assets](docs/TRACKED_ROBOT_HAND_ASSETS.md),
  [Arti-MANO adapter](docs/ARTIMANO_ADAPTER.md),
  [Wuji target](docs/WUJI_HAND2_BETA1_TARGET.md) /
  [中文 Wuji 目标手](docs/WUJI_HAND2_BETA1_TARGET.zh-CN.md),
  [Arti-MANO A–E evaluation](docs/GRAB_ARTIMANO_QUALITY_EXPERIMENT.md),
  [Wuji GRAB retargeting](docs/WUJI_HAND2_GRAB_RETARGETING.md),
  [Wuji continuity](docs/WUJI_CONTINUOUS_RETARGETING.md)
- Repository policy:
  [data/license policy](docs/LICENSE_AND_DATA_POLICY.md),
  [third-party asset policy](docs/THIRD_PARTY_ASSET_POLICY.md),
  [contributing](CONTRIBUTING.md),
  [third-party notices](THIRD_PARTY_NOTICES.md)

Detailed stage history and implementation notes are maintained in:

- [docs/DEVELOPMENT_LOG.md](docs/DEVELOPMENT_LOG.md)
- [docs/ROADMAP.md](docs/ROADMAP.md)
- [docs/stages/](docs/stages/)
- [solver feasibility note](docs/SOLVER_FEASIBILITY_RESTORATION.md)
- [Stage 16 reference-tracking PPO](docs/stages/STAGE16_REFERENCE_TRACKING_PPO.md)

## License

Repository code and documentation are released under the GNU General Public License v3.0; see
[LICENSE](LICENSE). Tracked third-party assets retain their upstream licenses and notices under
`third_party/robot_hands/`. GRAB, MANO/SMPL-X, ContactPose, ManipTrans, and other external
resources remain subject to their own terms. Read
[docs/LICENSE_AND_DATA_POLICY.md](docs/LICENSE_AND_DATA_POLICY.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before use.

## Acknowledgments

We thank:

- the authors of [TopoRetarget](https://toporetarget2026.github.io/TopoRetarget/);
- the GRAB, MANO/SMPL-X, and ContactPose authors and maintainers;
- the ManipTrans project and the upstream Arti-MANO asset contributors;
- the Wuji Hand2 upstream asset contributors identified in the tracked provenance manifests.

Preserve upstream attribution and comply with every dataset, model, code, and asset license.

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

Also cite each dataset, body model, target-hand asset, and upstream implementation used in your
experiment. The local paper copy is [docs/TopoRetarget.pdf](docs/TopoRetarget.pdf), and upstream
references are listed in [docs/UPSTREAM_REFERENCES.md](docs/UPSTREAM_REFERENCES.md).
