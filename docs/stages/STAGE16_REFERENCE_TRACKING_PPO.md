# Stage 16 — Reference-Tracking PPO

## Stage 16-B world wrist-and-finger engineering extension

Stage 16-A below is retained unchanged as the paper-oriented, base-relative,
finger-only profile (`paper_finger_only_base_relative_v1`). Stage 16-B is a
separate `ENGINEERING_EXTENSION` named `WORLD_WRIST_FINGER_TRACKING_PROTOCOL`.
It exports Stage-12's world wrist trajectory, models the wrist as a dynamic
free body driven by a finite Cartesian wrench, and appends its six residual
degrees of freedom to the preserved 20 finger residuals. It is neither the
paper's minimal RL controller nor a model of a physical robot arm.

Stage 16-B is now closed in MuJoCo with
`STAGE16B_ADAPTIVE_MULTI_HORIZON_ORACLE_PARTIAL`. B.0 direct 20 Hz world
references, B.1 finite-wrench wrist control, and B.1b per-trajectory
fixed-horizon controllability are complete. The shared, state-only adaptive
H1/H5/H10 B.1c oracle's selected 32x3 run passes `170650` in 20/20
deterministic frame-0 episodes at 0.119 cm position, 0.667 degrees rotation,
and 0.158 cm max-axis error. `170105` fails 20/20 at 80% progress with
3.406 cm position, 22.497 degrees rotation, and 5.194 cm max-axis error. Its
one permitted bounded 48x4 upgrade reaches 82.5% with 3.637 cm position,
18.192 degrees rotation, and 5.002 cm max-axis error, but still fails. The one
permitted global budget upgrade is consumed. No further MuJoCo CEM,
horizon, dynamics, or oracle budget is authorized.

The fixed statuses are `STAGE16B_SINGLE_CLIP_PPO_NOT_STARTED_GATE_BLOCKED`,
`STAGE16B_TWO_CLIP_PPO_NOT_STARTED`, `MUJOCO_PPO_TRAINING_DEFERRED`, and
`MUJOCO_CORRECTNESS_BACKEND_CLOSED`. This is not a PPO failure and does not
prove that `170105` is physically impossible. MuJoCo remains the correctness,
deterministic-regression, contact-diagnostic, action-replay, and visualization
backend. GPU-parallel training moves to Stage 16-C after independent Isaac Lab
platform, asset, semantic-parity, and PhysX-oracle qualification.

The direct sources, generated references, and bounded report suite are all
ignored under `.local/`. The preserved Stage-16A baseline is archived at
`.local/archive/stage16_controllability_failure_baseline_20260801T100413Z_aeb0995/`;
the Stage-16B.1c closeout is rooted at
`.local/reports/stage16b_adaptive_oracle_single_ppo/`.
The complete contract is [WORLD_WRIST_FINGER_TRACKING.md](../rl/WORLD_WRIST_FINGER_TRACKING.md).

Stage 16-C.0 platform qualification and C.1 asset migration are complete with
the limitations recorded below. C.2 `DirectRLEnv` is validated on real GPU
smokes. C3-0 reference/frame replay is now
`C3_REFERENCE_OR_FRAME_CONTRACT_VALIDATED` using derived canonical-URDF FK
targets, and C.3R2 contact readout is
`C3_CONTACT_READOUT_VALIDATED`. The generic D6 wrapper has zero live GPU tensor
joints, so the permitted explicit serial 3P+3R articulation fallback exposes
six GPU tensor joints and no real arm. Its PD baseline and both subsequently
qualified full-articulation computed-torque profiles fail. C3R4 proves that an
apparent MPC worker termination was a reporter `KeyError`; the worker completes
both clips after repair. Correct 120 Hz boundaries, live bias, a spectrally
bounded solver step, and a substep-affine model still fail independent 1/6-step
holdout and both 41-frame gates. That original-timing result remains
`C3_WRIST_ACTUATION_ARCHITECTURE_BLOCKED` historical evidence.

The user then explicitly authorized one shared reference retiming. C3R5 keeps
the 41 source keys and NPZ hashes immutable, materializes a factor-8 derived
321-sample reference at 20 Hz, and changes no gain, effort limit, action,
observation or gate. The shared `high_authority_bounded` explicit 3P+3R profile
passes both clips; C3-0 through C3-5, task-level contact causality, 26-D action
bases, reset/termination classification, and zero formal wrist/object rollout
writes pass. The current result is
`STAGE16C3_SEMANTIC_QUALIFICATION_VALIDATED`, with
`finite_virtual_6d_wrist_actuator_v1` active for the retimed task. C.4 is the
next mandatory gate. Its formal 128/512/1024/2048/4096 aggregate-contact runs
all exit clean and finite as `STAGE16C4_GPU_VECTOR_BACKEND_VALIDATED`; the
selected geometry is 4096 environments x rollout 16 with four shards. C.5 has
not run in this active goal; C.6/C.7 PPO remain unauthorized with zero
samples/checkpoints, and C.8/C.9 remain TODO.
MuJoCo and PhysX need not be bitwise identical, but MuJoCo oracle evidence can
never directly authorize an Isaac Lab policy run.

## Stage 16-C.0 Isaac Lab platform contract

C.0 is an `ENGINEERING_INFRASTRUCTURE` qualification, not a paper-method or
policy result. It freezes an isolated Python 3.11.15 environment, Isaac Sim
5.1.0, Isaac Lab `v2.3.2` at
`37ddf626871758333d6ed89cf64ad702aef127d0`, and PyTorch 2.7.0 with its CUDA
12.8 runtime. Qualification requires host compatibility, exact source and
package identity, Isaac Sim and Isaac Lab imports, a finite 1000-step
headless GPU-PhysX scene, an official task smoke, CUDA observations/actions,
and independent step/reset evidence from 128 truly parallel environments.
The optional viewer is a soft gate.

C.0 itself cannot migrate Wuji or HO-Cap assets, implement a custom `DirectRLEnv`,
define Stage-16 reward/action logic, run a PhysX oracle, or start PPO. The reusable
bootstrap never accepts an EULA; the verifier can set the official
`OMNI_KIT_ACCEPT_EULA=YES` value only when explicit authorization is recorded
and `--accept-eula` is supplied for that run.

The current result is `STAGE16C0_ISAACLAB_PLATFORM_VALIDATED_WITH_LIMITATIONS`.
The user explicitly authorized process-scoped EULA acceptance; every hard
runtime gate passed, while interactive viewer evidence remains unavailable.

## Stage 16-C.1 asset migration contract

C.1 imports the exact Wuji Hand2 Beta1 right-hand release source and preserves
a floating `r_wrist`, 20 revolute joints, 16 tracked links, and source limits.
High-poly upstream collision cooking exceeded the bounded runtime budget, so
the accepted strategy uses deterministic support-direction convex proxies
(21 bodies, at most 61 vertices each) while preserving upstream visual USD.
HO-Cap 170105 and 170650 preserve the original OBJ visual meshes and use one
uniform deterministic convex-hull proxy each (51/47 vertices). Both retain the
frozen nominal 0.05 kg mass, mesh-derived inertia, zero gravity, no ground, and
no support; physical provenance remains unresolved.

Real GPU PhysX smokes validate 20/20 joint response, 16/16 tracked-link
resolution, runtime limits within `1.2e-7 rad`, free-object stability, bounded
hand-object contact, CUDA tensors, 128 unique environment origins, and zero
subset-reset position error. C.1 does not implement reward, termination,
`DirectRLEnv`, oracle, or PPO. Headless numerical evidence is hard; interactive
visual review remains a soft limitation on this no-display host.

Stage 13 (additional dataset adapters), Stage 14 (robot plugin matrix), and Stage 15 (complete baseline matrix) are deliberately `DEFERRED`. This branch implements only the TopoRetarget reference-tracking PPO protocol from Appendix A.5.

## Stage 16.1–16.3 current closeout

The current status is `STAGE16_1_CONTROLLABILITY_BLOCKED`:

| Stage | Status | Evidence |
|---|---|---|
| 16.0 | `FUNCTIONAL_PIPELINE_COMPLETE` | Existing T1/T2/T3 functional run is frozen at 512 T3 samples; its 0% nominal/robust result is not a tracking qualification. |
| 16.1 | `STAGE16_1_CONTROLLABILITY_BLOCKED` | Stage-16.1a passes isolated dynamic-hand/kinematic-object PD tracking, but has no hand–object contact or proximity at frames 0/5/10 for either clip; free-object tracking crosses the unchanged 5 cm gate at frames 5–6. |
| 16.2 | `NOT_STARTED_GATE_BLOCKED` | Requires both clips to pass the Stage 16.1 controllability gate. |
| 16.3 | `NOT_STARTED_GATE_BLOCKED` | Requires both single-clip overfit gates. |

The Stage-16.1a baseline is frozen under `.local/archive/stage16_controllability_failure_baseline_20260801T060846Z_189b2f8/`; corrected A–E evidence is under `.local/reports/stage16_dynamic_coupling_v1_rerun1/`. An initial timestamp-alignment diagnostic pass is retained, unmodified, under `.local/reports/stage16_dynamic_coupling_v1/`; it is not used for the decision. No old checkpoint, Stage 7–12 artifact, or raw NAS data is modified.

## Stage-16.1a dynamic-coupling decision

The runner `scripts/rl/diagnose_stage16_dynamic_coupling.py` preserves formal termination while retaining full 41-frame diagnostic traces. It records physics/control contact traces, collision/proximity evidence, velocity-reset C0–C3, a cloned-state central-difference object-aware 20D residual oracle, and fixed-budget H=5/H=10 shooting. The oracle and shooting controller are engineering diagnostics, never policy results.

Both clips pass Step A (`q` RMSE 0.01544/0.01594 rad; link RMSE 0.789/0.865 mm). Step B finds no actual or expected proximity contact at frames 0/5/10 under any global preload candidate. At later static frames contact does exist, proving collision filters and object geoms are active, but it begins after the dynamic episode has already failed. The selected C3 hand-reference reset retains the best worst-clip progress under the frozen tie-breakers, but does not alter the frame-5/6 failure. The true object-aware oracle has full local rank 20, yet it also fails before final reach.

Therefore this implementation records `REFERENCE_DYNAMICAL_INFEASIBILITY` for the **current** fixed-base, 20D-finger, current-contact setup. It does not establish a paper-level property of the source motions or of PPO. Stage 16.2/16.3 must remain unstarted unless a separately authorized change to the protocol/scene establishes an early supporting contact without changing the frozen references or formal gates.

The current engineering simulator is MuJoCo 3.3.6 CPU with a per-clip collision mesh, a free object, zero gravity, and a wrist-relative frame. These are explicit engineering assumptions, not an author-exact simulator. Oracle output is only a controllability diagnostic; it must not be reported as PPO success.

The protocol is paper-exact where public: base-frame reference quantities, residual finger-joint action, the observation layout and offsets, Table 4 reward/termination, Table 5 randomization ranges, and Table 6 network/PPO values. The simulator, tracked-link list, axis offsets, action scale, gains, and unlisted PPO fields remain explicit assumptions in the Stage 16 ledger.

Run environment audit:

```bash
conda env create -f environment.stage16.yml
conda run -n toporetarget-rl python scripts/rl/audit_stage16_environment.py
conda run -n toporetarget-rl python scripts/rl/qualify_stage16_environment.py
```

Only a provenance-complete dynamic `Stage16ReferenceClip` is eligible for training. The implementation refuses static ContactPose samples. The currently installed MuJoCo backend is an isolated CPU correctness backend; it is never reported as the author-exact backend or as a 4096-environment reproduction.

Results, generated references, videos, checkpoints, reports, and build products are ignored under `.local/`.
The qualification command exercises E0--E4 only against a synthetic neutral Wuji asset reference;
it is a simulator/contract check, never a HOCap reference or policy result.

## Accepted HOCap functional protocol

For an accepted Stage-12 final, first materialize an ignored `RobotReferenceV2`,
the 20 Hz Stage-16 clip, and a derived local OBJ collision mesh. The raw HOCap
dataset remains read-only under its configured storage root.

```bash
conda run -n toporetarget-rl python scripts/rl/materialize_stage12_hocap_reference.py \
  --final-trajectory "$RUN/final/final_refinement_fast_exact_v2_r1/final_retarget.zarr" \
  --canonical "$RUN/canonical/canonical_hoi_v2.zarr" \
  --checkpoint-manifest "$RUN/checkpoints/final_refinement_fast_exact_v2_r1/manifest.json" \
  --robot-reference-output .local/stage16_reference_tracking_ppo/references/clip.robot_reference.zarr \
  --object-mesh-output .local/stage16_reference_tracking_ppo/objects/clip.obj \
  --report .local/reports/stage16_reference_tracking_ppo/accepted_reference_clip.json
conda run -n toporetarget-rl python scripts/rl/export_reference_clips.py \
  --robot-reference .local/stage16_reference_tracking_ppo/references/clip.robot_reference.zarr \
  --output .local/stage16_reference_tracking_ppo/references/clip.stage16.npz
```

`qualify_hocap_pd.py`, `train_hocap_reference_ppo.py`, and
`evaluate_hocap_reference_policy.py` run the bounded functional protocol.
They use the per-object derived collision mesh. Because these accepted inputs
are wrist-relative, the CPU runtime explicitly uses no synthetic ground,
zero gravity, and an explicitly disabled absolute-height termination; this is
an engineering assumption, not an author-exact simulator claim. Two accepted
clips demonstrate functionality but are not the paper's HOCap-32 evaluation.
