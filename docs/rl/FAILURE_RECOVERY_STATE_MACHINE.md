# Stage 16 bounded recovery

Every failure transition persists: phase, classified failure, evidence, attempt number, predefined fallback, repair, rerun scope, result, and remaining budget. Limits are three repairs per class, five reruns per phase, three backend switches, and twenty major repairs. Exhaustion escalates rather than looping.

Stage-specific budgets are now explicit in code:

- `Stage161RecoveryStateMachine`: at most 3 repairs per class, 5 reruns per phase, and 12 major transitions.
- `Stage162RecoveryStateMachine`: at most 3 repairs per class, 5 reruns per phase, and 16 major transitions.
- `Stage163RecoveryStateMachine`: at most 3 repairs per class, 5 reruns per phase, and 16 major transitions.
- `Stage161DynamicCouplingStateMachine`: ordered `STEP_A_PD → STEP_B_CONTACT → STEP_C_VELOCITY_RESET → STEP_D_OBJECT_ORACLE → STEP_E_DYNAMIC_FEASIBILITY → FINAL_REQUALIFICATION`, with the Stage-16.1 budget and no PPO transition.

The Stage 16.1 run classified the observed failures as `OBJECT_DYNAMICS_FAILURE` and
`ACTUATOR_OR_PD_FAILURE`; global object-mass recovery profiles 0.5, 1.0, and 5.0 kg were
executed in separate ignored run directories and did not change the failure location. The
result remains `STAGE16_1_CONTROLLABILITY_BLOCKED`; no PPO training was started behind a failed
controllability gate.

The current Stage-16.1a log is `.local/reports/stage16_dynamic_coupling_v1_rerun1/failure_transition_log.jsonl`. It recorded six bounded transitions. Step A passed, but Step B failed the pre-gate contact condition: both clips have zero actual and expected proximity contacts at static frames 0/5/10, while the formal free-object episode fails at frame 5 or 6. The classification is `REFERENCE_DYNAMICAL_INFEASIBILITY` for the current fixed-base/20D/contact setup. It is not relabeled as actuator failure or PPO failure.

GPU OOM follows `4096 -> 2048 -> 1024 -> 512 -> 256` with explicit shard accounting. PPO numerical failures roll back only to an atomic checkpoint and revalidate observation/reward/log probability first. Learning stalls require diagnostics before the one globally fixed fallback profile is attempted; no clip-specific reward, PD, or action scale is permitted.

## Stage 16-B bounded recovery

The separate world-wrist extension uses the same fail-closed principle. After
the retained wrist-control corrections, the authorized contact-aware MPC
extension audits mass/inertia/support, records substep contact impulse, and
compares the shared 12x2, 24x3, and 32x3 sequence-search budgets. The selected
32x3 H10 controller passes `170650`, but `170105` reaches 97.5% and fails
`FAILURE_OBJECT_POSITION`; W4 therefore remains `OBJECT_DYNAMICS_FAILURE`.

No clip-specific tuning, object pose write, physical-parameter selection by
tracking score, or PPO bypass is permitted. The complete selected transition
log is `.local/reports/stage16_world_wrist_finger/contact_mpc_formal_selected_20260801/failure_transition_log.jsonl`.
