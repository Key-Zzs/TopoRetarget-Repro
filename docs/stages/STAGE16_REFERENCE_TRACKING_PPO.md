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
H1/H5/H10 B.1c oracle's final 48x4 attempt passes `170650` in 20/20
deterministic frame-0 episodes at 0.250 cm position, 1.836 degrees rotation,
and 0.367 cm max-axis error. `170105` fails 20/20 at 82.5% progress with
3.637 cm position, 18.192 degrees rotation, and 5.002 cm max-axis error. The
one permitted global budget upgrade is consumed. No further MuJoCo CEM,
horizon, dynamics, or oracle budget is authorized.

The fixed statuses are `STAGE16B_SINGLE_CLIP_PPO_ENTRY_NOT_AUTHORIZED`,
`STAGE16B_SINGLE_CLIP_PPO_BLOCKED`, `STAGE16B_TWO_CLIP_PPO_BLOCKED`, and
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

Stage 16-C starts with C.0 platform qualification only. C.1 asset migration,
C.2 `DirectRLEnv`, C.3 semantic parity, C.4 vector benchmarks, C.5 PhysX
oracle, C.6/C.7 PPO, C.8 randomization, and C.9 comparison remain TODO.
MuJoCo and PhysX need not be bitwise identical, but MuJoCo oracle evidence can
never directly authorize an Isaac Lab policy run.

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
