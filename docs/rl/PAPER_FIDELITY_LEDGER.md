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
| Oracle gate | not specified | clone-only contact-aware H-by-26 sequence MPC at H=1/5/10 | ENGINEERING_DIAGNOSTIC | `oracle_evaluation.json` |

Stage 16-B preserves the Stage-16A ledger and is not a paper-fidelity upgrade.
Its authoritative result is `STAGE16B_BLOCKED_WITH_BOUNDED_EVIDENCE`; PPO did
not start because the 26D oracle gate failed.
