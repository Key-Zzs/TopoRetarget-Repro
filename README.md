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
export TOPORETARGET_OUTPUT=/path/to/toporetarget-output
```

Set local paths directly or start from
[configs/paths.example.yaml](configs/paths.example.yaml). `GRAB_ROOT` must
contain the licensed dataset; `MANO_MODEL_ROOT` must contain the licensed MANO
model files.

## Quick Start and Core Workflows

Complete [Setup](#setup) first. The sequence below starts with a small smoke
check, then moves through dataset preparation, geometric retargeting,
Stage16-D, evaluation, and replay. Raw licensed data is read-only; write
derived caches, reports, and HTML outside the repository, for example under
`$TOPORETARGET_OUTPUT`.

### 1. Environment entry and minimal smoke check

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

### 2. Dataset and reference preparation

Use one explicitly selected sequence at a time. Convert and inspect one HOI
sequence to create a manifest-bound canonical cache without resampling the
source sequence:

```bash
"$TOPORETARGET_PYTHON" -m toporetarget data convert \
  --dataset grab --sequence <sequence-id> --grab-root "$GRAB_ROOT" \
  --mano-model-root "$MANO_MODEL_ROOT" \
  --output "$TOPORETARGET_OUTPUT/<sequence-id>.zarr"
"$TOPORETARGET_PYTHON" -m toporetarget data inspect \
  "$TOPORETARGET_OUTPUT/<sequence-id>.zarr"
```

### 3. Core geometric retargeting

Inspect the exact options first, then use `plan-grab`, `run-grab`, `status`,
and `validate` against the same explicit sequence/window and output root. The
workflow is resumable and does not scan or mutate unrelated source data.

```bash
"$TOPORETARGET_PYTHON" -m toporetarget workflow plan-grab --help
"$TOPORETARGET_PYTHON" -m toporetarget workflow run-grab --help
"$TOPORETARGET_PYTHON" -m toporetarget workflow status --help
"$TOPORETARGET_PYTHON" -m toporetarget workflow validate --help
"$TOPORETARGET_PYTHON" -m toporetarget geometry --help
```

### 4. Stage16-D causal PPO entry

The Stage16-D causal PPO workflow is documented in [Physics-correction
PPO](docs/rl/PHYSICS_CORRECTION_PPO.md) and [Stage 16-D
physics-consistent retargeting](docs/stages/STAGE16D_PHYSICS_CONSISTENT_RETARGETING.md).
It remains a causal reference-tracking baseline under the frozen simplified
zero-gravity, no-support Isaac/PhysX contract: no guidance forces, support,
attachments, hidden object controller, or rollout-time object-state or
wrist-root writes. It does not claim full-gravity or real-world physical
validation.

The follow-on physical bootstrap defines Contact-ready RSI V2, source-support
feasibility, and a fail-closed P3 entry decision. It is diagnostic-only and
does not start PPO or a gravity/friction curriculum; see [Stage 16 Physical
Bootstrap](docs/stages/STAGE16_PHYSICAL_BOOTSTRAP.md).

### 5. Evaluation, replay, and visualization

Geometry inspection is read-only; a frozen benchmark follows the explicit
`inspect-datasets -> select -> freeze -> run -> evaluate` state machine:

```bash
"$TOPORETARGET_PYTHON" -m toporetarget benchmark inspect-datasets --help
"$TOPORETARGET_PYTHON" -m toporetarget benchmark select --help
"$TOPORETARGET_PYTHON" -m toporetarget benchmark freeze --help
"$TOPORETARGET_PYTHON" -m toporetarget benchmark run --help
"$TOPORETARGET_PYTHON" -m toporetarget benchmark evaluate --help
```

Inspect an existing Isaac Lab trace. Replay is diagnostic only: it does not
retrain PPO, alter the trace, or create a new physics qualification.

```bash
conda run -n toporetarget-isaaclab env OMNI_KIT_ACCEPT_EULA=YES \
  python scripts/rl/isaaclab/replay_stage16d_simulation_trace.py --help
```

The project generates self-contained browser HTML for source, warm-start, and
final meshes; interaction graphs; contact and collision diagnostics; continuity;
and provenance. Use the visualization commands emitted by the selected
pipeline manifest, then inspect the generated HTML in a browser.

### 6. Further reproduction

The main offline pipeline is documented in [configs/README.md](configs/README.md)
and the CLI help. For the full argument contracts and acceptance boundaries,
see [workflow resume and provenance](docs/WORKFLOW_RESUME_AND_PROVENANCE.md) and
[the Isaac Lab direct environment contract](docs/rl/ISAACLAB_DIRECT_RL_ENV.md).
For a paper-fidelity check, run:

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
tracking together with versioned contact rewards. **Aggregate V3 is the current
stable/default contact baseline** (`aggregate_v3`). **Strict Per-Finger V4 is
experimental and opt-in** (`strict_per_finger_v4`); it instead uses
`SourcePerFingerContactEvidenceV1`: only source-confirmed or
persistent-confirmed MANO/object contact for a named finger requires that same
Wuji distal/tip body to contact the active object. Probable, transition,
proximity-only, no-contact, and ambiguous source states are not mandatory V4
contact semantics.

V4 normalizes independent named-tip rewards by the number of source-required
fingers. A large force from another finger therefore cannot credit a missing
required finger or change the total reward scale merely because the source
requires more fingers. The reward reads only current filtered PhysX
named-tip-to-active-object pair force and never directly controls the object.
Its shared per-tip force scale is frozen from exact V1 Formal20 pair-force
telemetry before PPO.

The completed Stage16-D milestone is a physically causal reference-tracking
baseline under a frozen simplified **zero-gravity, no-support** Isaac/PhysX
contract. It has no external object guidance and no rollout-time object-state
or wrist-root writes. This is not physically realistic, real-world calibrated,
or full-gravity validation. New configurations use the stable default:

```yaml
reward:
  contact:
    mode: aggregate_v3
```

To explicitly opt into the experimental objective:

```yaml
reward:
  contact:
    mode: strict_per_finger_v4
```

Phase-specific terminal-dynamics attribution and detailed results are recorded
in stage and RL documentation, with machine-readable artifacts kept in ignored
local storage.

## Documentation map

- [Roadmap](docs/ROADMAP.md) — current causal research route and future lanes.
- [Stage 16-D physics-consistent retargeting](docs/stages/STAGE16D_PHYSICS_CONSISTENT_RETARGETING.md)
  — physics scope, provenance, and qualification boundary.
- [Stage16-D causal zero-g milestone](docs/stages/STAGE16D_CAUSAL_ZERO_G_MILESTONE.md)
  — frozen scope, stable/default V3, experimental V4, and the next physical stage.
- [Stage 16 Physical Bootstrap](docs/stages/STAGE16_PHYSICAL_BOOTSTRAP.md)
  — P0/P1/P2 contracts, safe-bank boundary, and P3 entry gates.
- [Terminal dynamics attribution](docs/stages/STAGE16D_PHASE1_TERMINAL_DYNAMICS.md)
  — Phase 1 method and conclusions.
- [PPO-26D reference tracking](docs/rl/REFERENCE_TRACKING_PPO_26D.md) — action,
  observation, RSI, reward, and gate contracts.
- [Physics-correction PPO](docs/rl/PHYSICS_CORRECTION_PPO.md) — causal training
  boundary and decision tree.
- [Reference-gated contact reward](docs/rl/REFERENCE_GATED_CONTACT_REWARD.md)
  — V3 contact signal and causal boundary.
- [Strict per-finger contact reward](docs/rl/STRICT_PER_FINGER_CONTACT_REWARD.md)
  — V4 source-confirmed contact semantics and independent-finger contract.
- [Source contact semantics](docs/rl/SOURCE_CONTACT_SEMANTICS.md) — raw
  MANO/object evidence and frozen factor-eight runtime mapping.
- [Evaluation Suite V2](docs/rl/EVALUATION_SUITE_V2.md) — shared metric and
  success contract.
- [Paper fidelity and engineering adaptations](docs/PAPER_FIDELITY.md) — what
  follows the paper and what is explicitly adapted.

## README document policy

README files are stable project entry documents; experiment logs and
run-specific metrics live outside README, in stage documentation and local
machine-readable reports.

## Acknowledgements

This repository is an independent reproduction and engineering extension of
the ideas presented in [*TopoRetarget: Interaction-Preserving Retargeting for
Dexterous Manipulation*](https://arxiv.org/abs/2606.16272). We thank the
original authors for that research contribution.

We also acknowledge the upstream projects that provide core foundations used
here, including MANO/SMPL-X, PyTorch, Trimesh, python-fcl, MuJoCo, and NVIDIA
Isaac Sim/Isaac Lab, as well as the authors and maintainers of the external
datasets used by the workflows. The tracked Arti-MANO snapshot is sourced from
[ManipTrans](https://github.com/ManipTrans/ManipTrans), and the tracked Wuji
Hand2 Beta1 subset is sourced from
[wuji-description](https://github.com/wuji-technology/wuji-description).

Third-party datasets, body models, robot assets, and software remain subject to
their respective licenses and terms of use; this repository's license does not
automatically relicense those materials.

## Citation

If you use the reproduced TopoRetarget method, please cite the original paper:

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

If you use this repository's engineering extensions, evaluation tools, or
simulation infrastructure, please also cite the repository:

```bibtex
@software{keyzzs_toporetarget_repro_2026,
  author = {{Key-Zzs}},
  title  = {TopoRetarget-Repro},
  url    = {https://github.com/Key-Zzs/TopoRetarget-Repro},
  year   = {2026}
}
```

The repository owner identity in this software citation is taken from the
repository metadata; no personal author name or DOI is inferred. When using
external datasets, body models, robot assets, or upstream software, also cite
those projects according to their own instructions. The local paper copy is
[docs/TopoRetarget.pdf](docs/TopoRetarget.pdf), and upstream asset provenance is
recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

The repository code and documentation are released under the GNU General Public
License v3.0; see [LICENSE](LICENSE). Tracked third-party assets retain their
upstream licenses and notices under `third_party/robot_hands/`. External GRAB,
MANO/SMPL-X, and other dataset/model resources are not redistributed here and
remain subject to their own terms. See [docs/LICENSE_AND_DATA_POLICY.md](docs/LICENSE_AND_DATA_POLICY.md)
and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before using external
resources.
