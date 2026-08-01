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

The separate world-wrist extension uses the same fail-closed principle.
`Stage16BAdaptiveOracleStateMachine` limits every failure class to three
repairs, formal reruns to five, major transitions to twelve, and CEM budget
upgrades to one. The selected 32x3 adaptive H1/H5/H10 run passes `170650`
20/20 but fails `170105` 20/20 at 80% progress. The bounded 48x4 `170105`
upgrade reaches 82.5% and still fails at 5.002 cm axis error. The single
global budget upgrade is consumed. The final class is
`CEM_CONTACT_MODE_MISS` under the frozen engineering simulator; recovery is
exhausted and the lane closes as partial rather than looping.

No clip-specific tuning, object pose write, physical-parameter selection by
tracking score, additional horizon/budget, or PPO bypass is permitted. PPO
remains `NOT_STARTED_GATE_BLOCKED`; the PPO recovery log is empty because no
PPO transition occurred. This is not a training failure. The final oracle
log is `.local/reports/stage16b_adaptive_oracle_single_ppo/oracle_failure_transitions.jsonl`.

## Stage 16-C.0 bounded recovery

`Stage16C0RecoveryStateMachine` limits each installation/runtime failure class
to three repairs, all retries to four, installation-method switches to two,
and major transitions to sixteen. Its failure classes separate host/driver,
glibc, Python, Torch/CUDA, Isaac Sim import, Isaac Lab import, EULA,
network/cache, headless rendering, and GPU PhysX failures. It never upgrades a
driver, kernel, system glibc, or system CUDA, and it never converts a CPU
fallback into a GPU pass.

The current transition is `EULA_REQUIRED -> complete_static_audit_only` with
result `ISAACLAB_EULA_ACCEPTANCE_REQUIRED`. It consumes no retry or install
method switch: no explicit user authorization is recorded, so the verifier
stops before importing Isaac Sim. C.0 is therefore blocked, C.1 is not
authorized, and runtime success is not fabricated.
