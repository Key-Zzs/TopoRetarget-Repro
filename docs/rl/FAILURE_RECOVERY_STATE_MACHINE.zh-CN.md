# Stage 16-C failure-recovery 状态机

```
冻结 C2/C3 输入
  -> C3-0 reference/FK
  -> object-centric contact readout
  -> full-articulation computed torque
  -> bounded preview (TVLQR/MPC)
  -> wrist gate 通过？ -- 否 --> C3 BLOCKED -> C4/C5 NOT RUN -> PPO NOT AUTHORIZED
                       是
                         -> contact-momentum causality
                         -> 完整 C3 -> C4 -> C5
```

原始时间轴在 bounded preview 后 fail-closed：两档 computed-torque、独立 1/6-step
local-model holdout 与两条 41-frame MPC gate 均失败。MPC worker 的早期“退出”已确认是
reporter `KeyError` 误报，不是 CUDA/PhysX 退出。其后经单独授权的共享 factor-8 retiming
保持 source hash/key 不变，完成 C3/C4；这些结果仍不授权 C5 Oracle 或 PPO。

## Stage 16-C.5A failure-recovery state machine

`Stage16C5ARecoveryStateMachine` 显式限制 C.5A repair budget：每个 failure class 最多三次
repair、每 phase 最多五次 rerun、最多一次 replication-method switch、最多 24 次 major transition。
input/hash drift、candidate setup 之外的 write、execution-rollout direct state write，以及 natural
baseline hard-cap failure 均 fail-close。

唯一允许的 fallback 为 `deterministic_history_replay_v1`，且仅在 no-clone baseline 通过、出现
tensor-clone contact mismatch 后可用。它将 candidate ID reset 到 frame zero，再推进普通 20 Hz
control action；replay 中途绝不写 object state。本次 baseline 在 O1 之前已经失败，因此 fallback
不具备资格。

本次只使用一次有界 repair：1-candidate O0 peer check 实际没有 peer，但旧脚本仍以 peer 比较。
修复 rerun 正确处理 no-peer，同时不隐藏原始 artifact。随后
`PHYSX_REPLICATION_BASELINE_NONDETERMINISM` transition 终止 O1、fallback 和 benchmark；它不是可
用 tolerance tuning 恢复的 transition。

有序 transition 位于被忽略的
`.local/reports/stage16c5a_state_replication/failure_transitions.jsonl`。
