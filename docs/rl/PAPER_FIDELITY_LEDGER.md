# Stage 16 paper-fidelity ledger

| Item | Paper value | Implemented mapping | Classification | Evidence |
| --- | --- | --- | --- | --- |
| Reference frame | robot base | `Stage16ReferenceClip.object_pose_base_ref` | PAPER_EXACT | Appendix A.5, p. 13 |
| Action | `q_ref + residual` | `residual_target` | PAPER_EXACT | A.5.1, p. 13 |
| Observation | q, qdot, previous action, axes, refs [0,1,3,5] | `build_observation` | PAPER_EXACT | A.5.2, p. 13 |
| Reset | uniform reference start frame | environment reset contract | PAPER_EXACT | A.5.3, p. 13 |

## Qualification boundary

`frame0_deterministic_eval_v1` is the formal engineering evaluation protocol for this branch:
all episodes reset at `reference_index=0`, use the deterministic actor mean, and retain failed
episodes. Stage 16.1 is blocked because both approved clips fail the shared free-object
controllability gate under zero-residual and object-blind oracle diagnostics. The old 512-sample
functional T3 run is frozen separately and is not evidence of tracking qualification.
| Reward | Table 4 | `paper_literal_reward_v1` | PAPER_EXACT | Table 4, p. 14 |
| Termination | Table 4 | `classify_termination` | PAPER_EXACT | Table 4, p. 14 |
| DR | Table 5 | `DomainRandomizationConfig` | PAPER_EXACT | Table 5, p. 15 |
| PPO | Table 6 | `PPOConfig`, `ActorCritic` | PAPER_EXACT | Table 6, p. 16 |
| Tracked links | not listed | `tracked_link_profile_v1` | ENGINEERING_ASSUMPTION | A.5.4 omits identities |
| Axis-point offsets | not listed | 6 endpoints at 0.05 m | ENGINEERING_ASSUMPTION | A.5.2/A.5.4 omit scale |
| Action scale, PD | not listed | fixed qualification candidates | ENGINEERING_ASSUMPTION | A.5.1 omits values |
| PPO clip/value/grad | not listed | `0.2/0.5/1.0` | ENGINEERING_ASSUMPTION | A.5.6 omits values |
| Simulator | not listed | MuJoCo 3.3.6 CPU correctness backend | ENGINEERING_ASSUMPTION | Appendix A.5 omits backend |
| Pen-Spin clips | private, 32 | unavailable, fail closed | UNRESOLVED | A.4, p. 13 |

`PAPER_EXACT` means the public statement is implemented literally; it does not imply author-code, asset, data, or result equivalence.

## Stage 16-B world wrist-and-finger ledger

| Item | Paper value | Implemented mapping | Classification | Evidence |
| --- | --- | --- | --- | --- |
| World wrist reference | not specified | `WorldWristFingerReferenceV1`, direct Stage-12 `base_pose_scene` export | ENGINEERING_EXTENSION | `scripts/rl/export_stage16_world_wrist_reference.py` |
| Wrist actuation | not specified | `CartesianWristImpedanceController`, finite world wrench | ENGINEERING_EXTENSION | `world_wrist_backend.py` |
| 26D action | not specified | 6D local wrist residual + original 20D finger residual | ENGINEERING_EXTENSION | `WristFingerActionScaleV1` |
| World/relative features | not specified | `WorldWristObservationContractV1` (764D) | ENGINEERING_EXTENSION | `REFERENCE_TRACKING_MDP.md` |
| Dynamic scene | not specified | MuJoCo 3.3.6, zero gravity, no ground, free object | ENGINEERING_ASSUMPTION | `wrist_model_validation.json` |
| Adaptive oracle gate | not specified | shared state-only terminal-contracted H1/H5/H10 CEM with gate-first selection | ENGINEERING_DIAGNOSTIC | `stage16b_adaptive_oracle_single_ppo/adaptive_oracle_evaluation.json` |
| Nominal simulator profile | not specified | `world_wrist_freebody_nominal_v1`; 0.05 kg, mesh inertia, zero gravity/support/damping | ENGINEERING_ASSUMPTION | `nominal_dynamics_profile.json` |
| Per-clip PPO | Appendix A.5 does not specify this 26D extension | gate not authorized; 0 samples and no checkpoints | NOT_STARTED_GATE_BLOCKED | `ppo_contract.json` |
| MuJoCo backend role | not specified | correctness, deterministic regression, contact diagnostics, action replay, visualization | ENGINEERING_INFRASTRUCTURE | Stage 16-B.1c closeout |
| Isaac Lab GPU lane | not specified | independent Stage 16-C platform/PhysX qualification before PPO | ENGINEERING_EXTENSION | Stage 16-C roadmap |

Stage 16-B preserves the Stage-16A ledger and is not a paper-fidelity upgrade.
Its authoritative result is
`STAGE16B_ADAPTIVE_MULTI_HORIZON_ORACLE_PARTIAL`; `170650` passes while
`170105` does not under the frozen MuJoCo model and bounded search. PPO did not
start. This MuJoCo result cannot authorize Isaac Lab PPO, and no PhysX result
is claimed.

## Stage 16-C.0 Isaac Lab platform ledger

| Item | Paper value | Implemented mapping | Classification | Evidence |
| --- | --- | --- | --- | --- |
| GPU simulator platform | not specified | Isaac Sim 5.1.0 + Isaac Lab v2.3.2 | ENGINEERING_INFRASTRUCTURE | `isaaclab_platform.yaml` |
| Python/Torch runtime | not specified | Python 3.11.15, Torch 2.7.0 cu128 | ENGINEERING_INFRASTRUCTURE | environment manifests |
| GPU PhysX smoke | not specified | finite official headless platform smoke only | ENGINEERING_DIAGNOSTIC | Stage 16-C.0 report bundle |
| 128-env vector smoke | not specified | official task, CUDA tensors, independent actions/resets | ENGINEERING_DIAGNOSTIC | `vector_env_benchmark.json` |
| Stage-16 assets/task/oracle/PPO | not specified | prohibited during C.0 | NOT_STARTED_SCOPE_BLOCKED | frozen C.0 scope |

C.0 is not a paper-fidelity upgrade and is not evidence that the Stage-16
task is controllable in PhysX. It can authorize only C.1 asset migration after
all hard platform gates pass; it cannot authorize PPO. The current C.0 result
is `STAGE16C0_ISAACLAB_PLATFORM_BLOCKED` because explicit NVIDIA EULA
authorization is not recorded; C.1 remains unauthorized.
