# Stage 16-C.5A state-replication gate

## 当前收口

state-replication gate 的状态为
`STAGE16C5A_BLOCKED_PHYSX_REPLICATION_BASELINE_NONDETERMINISM`。这是 fail-closed
诊断，不是 PhysX Oracle、CEM、policy 或 PPO 结果。没有运行 full Oracle episode、正式
20-episode evaluation，没有产生 checkpoint 或 PPO sample。

冻结输入是 C3R5 factor-8 派生 reference：两条原始 source NPZ hash 与全部 41 个 source key
保持不变，runtime 为 321 samples、20 Hz。task 仍为 26 DoF/action、764 observation，physics 为
120 Hz、decimation 6。被忽略的 C.5A report bundle 中的 `frozen_inputs.json` 记录精确
source/config/asset hash 与 archive 路径。

## Candidate-state contract

`Stage16C5CandidateStateV1` 显式捕获每环境 simulation state（robot joint/root 与两个 object
root）、task/reference index、action history、controller target/residual、saturation/termination
buffer 与 environment origin。capture 校验 field existence、shape、discrete dtype、device 及所有
floating value finite。restore 按 destination env origin rebase world position，candidate setup
期间仅写 candidate ID。formal execution-rollout 禁止 direct wrist/object state write，并独立审计。

Isaac Lab 没有受支持的 API 用于恢复 PhysX solver warm-start、contact-manifold、friction-patch
或 internal constraint cache。这些不可复制状态被显式列出，绝不假定可 clone。

## O0 candidate-pool evidence

独立 CUDA process 验证 1、32、96、144 candidates，并使用独立 execution environment。每个接受的
run 均具有 unique environment origin、finite state tensor、保留的 clip/reference index、subset-reset
isolation、仅 candidate setup write，以及 0 formal execution-rollout write。future-only schedule
为 96 = three horizons x 32，升级为 144 = three horizons x 48；它只做 allocation，不实现 CEM 或
candidate scoring。

最初的 1-candidate smoke 错将不存在的 peer 标为 contaminated。原 partial artifact 被保留；一次
有界 bookkeeping repair 不再将没有 peer 当作 peer comparison，并在 `failure_transitions.jsonl` 中
同时记录原件和修复件。

## Baseline 与 stop rule

在 snapshot/restore 之前，natural no-clone baseline 对两条 clip 的 pre-contact、contact-onset、
sustained-contact、post-contact 各运行 20 trials（每 metric 160 samples）。规则为
`max(fixed_floor, 10 * baseline_p99)` 并带 hard cap。测得的 global tolerance 在 object
position/orientation、joint position、linear/angular velocity 与 reward 上超过 cap；精确数值在
`replication_noise_floor.json` 中冻结。

因此 O1 tensor-clone qualification、`deterministic_history_replay_v1`、超出 O0 的 candidate
independence 和 resource benchmark 均 **未运行**。fallback 只允许在 baseline 通过、但 tensor-clone
contact mismatch 时使用；不能用来规避此次失败，也没有放宽 tolerance。

## Gate boundary

C5B 需要 C5A O0/O1 都通过，本次未获授权；C6/PPO 也仍未获授权。未来只能先在同一冻结输入与
hard cap 下复现通过的 natural baseline，再对全部 phase 单独验证 tensor clone，之后才能考虑
有界 fallback 或 C5B。

machine-local evidence 位于被 Git 忽略的
`.local/reports/stage16c5a_state_replication/`。
