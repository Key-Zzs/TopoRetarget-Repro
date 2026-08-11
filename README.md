# TopoRetarget-Repro

[中文 README](README.zh-CN.md)

TopoRetarget-Repro is an independent, paper-traceable reproduction of
[*TopoRetarget: Interaction-Preserving Retargeting for Dexterous Manipulation*](https://arxiv.org/abs/2606.16272).
It turns hand-object motion into auditable dexterous-hand references through a
canonical HOI contract, MANO semantic conversion, target-hand kinematics,
geometry/SDF processing, interaction-aware refinement, validation, and
manifest-bound export.

## Research goal and boundary

The project investigates physically feasible dexterous-hand simulation while
preserving the causal chain:

```text
robot action -> hand-object contact -> object dynamics
```

The active causal lane uses PPO-26D physics correction. It does not use object
guidance forces, hidden object controllers, object-pose or velocity writes,
attachments, or suction during rollout. An eventual H2R assisted-data lane is
separate from this main causal solution and must label its outputs
`assisted=true` and `causal_physics=false`.

This repository is an engineering reproduction, not a claim of author-exact,
full-dataset, real-time, hardware-control, or vendor-supported reproduction.

## Method overview

```text
licensed HOI data
  -> canonical HOI sequence and coordinate conventions
  -> MANO / target-hand semantic conversion
  -> interaction-aware kinematic retargeting
  -> geometry and contact validation
  -> versioned robot reference export
  -> Isaac Lab causal physics correction and evaluation
```

The core contracts are [HOI data](docs/HOI_DATA_INTERFACE.md),
[coordinate conventions](docs/COORDINATE_CONVENTIONS.md), and the
[robot-hand target contract](docs/ROBOT_HAND_TARGET_CONTRACT.md).

## Supported data and hands

| Dataset | Adapter | Notes |
| --- | --- | --- |
| GRAB | Supported | Dynamic hand-object sequences |
| DexYCB | Supported | Native PCA45 and subject-shape routing |
| OakInk | Supported | Native hand vertices/joints and object transforms |
| HO-Cap | Supported | PCA45, subject shape, and object pose |
| ContactPose | Supported | Static one-frame conversion |
| ARCTIC, OakInk2, TACO | Planned | Not yet supported |

| Target hand | Kinematics | Retargeting | Collision | Simulation/RL |
| --- | --- | --- | --- | --- |
| Arti-MANO | Supported | Supported | Supported | Not automatically qualified |
| Wuji Hand2 Beta1 | Supported | Supported | Supported | Offline references and causal-physics lane |
| Generic URDF/MJCF | Import foundation | Manifest required | Profile required | Not automatically qualified |

External datasets and MANO/SMPL-X models are not redistributed. Keep licensed
inputs, models, generated data, caches, and local runs outside version control.

## Setup

The general workflow uses Python `>=3.10,<3.14`; Python 3.12 is the maintained
local setup. Isaac Lab uses its separate frozen environment described in the
[Isaac Lab direct environment contract](docs/rl/ISAACLAB_DIRECT_RL_ENV.md).

```bash
conda create -n topo-retarget python=3.12 -y
conda activate topo-retarget
python -m pip install -U pip
python -m pip install -e ".[dev,cache,viz,grab,robot,geometry,retarget]"

export GRAB_ROOT=/path/to/GRAB
export MANO_MODEL_ROOT=/path/to/body_models/mano
export PYTHONNOUSERSITE=1
export PYTHONPATH=src
export TOPORETARGET_PYTHON="${CONDA_PREFIX}/bin/python"
```

Set local paths directly or start from
[configs/paths.example.yaml](configs/paths.example.yaml). `GRAB_ROOT` must
contain the licensed dataset; `MANO_MODEL_ROOT` must contain the licensed MANO
model files.

## Quick start and reproduction

Inspect the available commands and validate the shipped target assets:

```bash
"$TOPORETARGET_PYTHON" -m toporetarget --help
"$TOPORETARGET_PYTHON" -m toporetarget doctor paper
"$TOPORETARGET_PYTHON" -m toporetarget robots list
"$TOPORETARGET_PYTHON" -m toporetarget robots validate artimano_rh \
  --asset-root third_party/robot_hands/artimano
"$TOPORETARGET_PYTHON" -m toporetarget robots validate wuji_hand2_beta1_rh \
  --asset-root third_party/robot_hands/wuji_hand2_beta1
```

The main offline pipeline is documented in [configs/README.md](configs/README.md)
and the CLI help. It preserves source data, creates manifest-bound derived
outputs, and keeps human acceptance as an explicit boundary. For a paper-fidelity
check, run:

```bash
"$TOPORETARGET_PYTHON" scripts/check_paper_fidelity.py
```

## Visualization

The project generates self-contained browser HTML for source, warm-start, and
final meshes; interaction graphs; contact and collision diagnostics; continuity;
and provenance. Use the visualization commands emitted by the selected pipeline
manifest, then inspect the generated HTML in a browser. Replay of a saved Isaac
Lab trace is diagnostic visualization only; it does not create a new physical
qualification.

## Evaluation

The common evaluation entry point is [Evaluation Suite V2](docs/rl/EVALUATION_SUITE_V2.md).
It reports object rotation and translation tracking, retargeted-hand joint and
fingertip tracking, plus separate kinematic, physics, and qualified success
rates. Trajectory metrics use a common world/env frame with the environment
origin removed; legacy metrics remain available for comparison but are not
silently redefined.

The Stage 16-D causal PPO pipeline supports reference pose and object-twist
tracking together with a reference-gated contact-consistency reward. The latter
uses reference Wuji fingertip proximity and current PhysX fingertip-to-active-
object pair force only during policy optimization; its shared force scale is
frozen from exact V1 Formal20 pair-force traces, and it never directly controls
the object.

Phase-specific terminal-dynamics attribution and detailed results are recorded
in stage and RL documentation, with machine-readable artifacts kept in ignored
local storage.

## Documentation map

- [Roadmap](docs/ROADMAP.md) — current causal research route and future lanes.
- [Stage 16-D physics-consistent retargeting](docs/stages/STAGE16D_PHYSICS_CONSISTENT_RETARGETING.md)
  — physics scope, provenance, and qualification boundary.
- [Terminal dynamics attribution](docs/stages/STAGE16D_PHASE1_TERMINAL_DYNAMICS.md)
  — Phase 1 method and conclusions.
- [PPO-26D reference tracking](docs/rl/REFERENCE_TRACKING_PPO_26D.md) — action,
  observation, RSI, reward, and gate contracts.
- [Physics-correction PPO](docs/rl/PHYSICS_CORRECTION_PPO.md) — causal training
  boundary and decision tree.
- [Reference-gated contact reward](docs/rl/REFERENCE_GATED_CONTACT_REWARD.md)
  — V3 contact signal and causal boundary.
- [Evaluation Suite V2](docs/rl/EVALUATION_SUITE_V2.md) — shared metric and
  success contract.
- [Paper fidelity and engineering adaptations](docs/PAPER_FIDELITY.md) — what
  follows the paper and what is explicitly adapted.

## README document policy

README files are stable project entry documents. Experiment logs, checkpoint
receipts, per-stage metrics, commit-specific status, and runtime reports are
intentionally excluded. Detailed results live in stage documentation and local
machine-readable reports.

## License

See [LICENSE](LICENSE). Respect the licenses of upstream datasets, models,
robot assets, and dependencies.

## Citation

Please cite the original TopoRetarget paper for the method. Cite this repository
according to its release metadata when using its implementation or derivative
artifacts.
