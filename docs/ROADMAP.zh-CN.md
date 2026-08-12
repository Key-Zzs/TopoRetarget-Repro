# 路线图

本路线图描述因果科研路线，不是实验流水账。详细结果和 receipt 进入 stage/RL 文档与忽略的本地
machine-readable reports。

## 基础能力

稳定基础包括规范 HOI 数据、语义 MANO 转换、目标手运动学、交互感知重定向、几何/SDF 验证、
绑定 manifest 的导出以及浏览器证据。当前两个目标手路线是 Arti-MANO 与 Wuji Hand2 Beta1。

## Stage 16-D 因果物理一致重定向

因果合同保持为：

```text
robot action -> hand-object contact -> object dynamics
```

main causal lane 禁止 object guidance force、hidden controller、object-state correction、
attachment 和 suction。

### Phase 1 — Terminal Drift / Support / RSI Attribution（VALIDATED）

使用冻结 PPO checkpoint 和正式 frame-zero episodes 确定 terminal drift 的来源。审计
reference terminal twist、actual/residual object twist、contact impulse/loss、zero-gravity
persistence、source/support metadata 和 RSI implementation/state quality。本阶段不训练 policy。

### Phase 2 — Evaluation Suite V2（VALIDATED）

冻结一个可加的统一评价合同，用于 single-clip PPO、Multi-Clip PPO、未来 adapter 和 physical
curriculum。它报告 `E_r`、`E_t`、`E_j`、`E_ft`、`SR_kinematic`、`SR_physics` 和
`SR_qualified`，同时以原名保留 legacy metrics。用该合同重新评价现有两条冻结 frame-zero baseline。

### Phase 2.5 — Reference Kinematics V2（VALIDATED）

将 41-key spatial specification、timestamps、world-frame linear/angular twist 与 terminal
semantics 冻结到单一 V2 artifact。V2 validation 证明仅一次的 factor-eight reference resampling
保持物理时间，且不使用 runtime re-timing shim。

### Phase 2.6 — V2 Evaluation and Entry Gate（VALIDATED）

在 V2 下重评冻结 baseline 与 Phase 1-R attribution。该 gate 将可信 reference target 与 policy
tracking error 分开，只授权有界的 `hocap_170650` Reward V2 experiment。

### Phase 1-R — V2 Attribution Rerun（COMPLETE）

terminal residual/contact attribution rerun 已完成。它是 evaluation gate，不是 policy-training phase。

### Object Twist Reward V2（COMPLETED / PARTIAL）

仅在 Phase 1–2 证明 reference twist 是可信 target，且 residual object dynamics 是重要 terminal
failure 后，才对 PPO tracking reward 版本化，加入 object linear-velocity 和 angular-velocity
tracking。本阶段从一条 causal single-clip 的 retraining/visualization/evaluation 开始；不包含
contact reward、external guidance 或 curriculum。

Reference Kinematics V2 与 Phase 1-R attribution 已通过 entry gates。获授权的
`hocap_170650` Reward V2 P1 probe 在第一个 1,048,576 sample gate 停止：相对冻结的
V1 4M baseline，terminal contact 与 stability 发生回退。该结果不授权继续到 4M/16M，
也不授权扩展 reward 或 physics contract。

### Reference-Gated Contact Reward V3（PARTIAL）

Reward V3 是当前唯一的因果单变量实验：

```text
Reward V3 = Reward V2 + reference-gated fingertip-to-active-object contact reward
```

它保留 Reference Kinematics V2、764-D observation、26-D action、physics、controller 和 PPO
hyperparameters。mask 只由 Wuji distal-root 到 visual object surface 的 reference proximity
决定；actual signal 严格是当前经 filter 的 fingertip--active-object PhysX pair force。不加入
contact-loss termination、terminal/penetration reward、guidance 或 physics curriculum。signal
qualification 采用 fail-closed：历史 aggregate force telemetry 不能替代精确 pair force。

两个 clip 的精确 V1 Formal20 pair-force re-export 均已验证，并以其 pooled
positive-contact median 一次性冻结共享 V3 force scale。有界 V3 结果为 partial：
`hocap_170105` 的 Formal20 qualified success 从 0/20 提升至 19/20，且
free-flight re-catch 减少；`hocap_170650` 从 14/20 提升至 16/20，但
free-flight re-catch 仍存在。这不授权 multi-clip PPO、reward-contract 扩展、
contact-loss termination 或 physics curriculum。下一步是审查冻结的
contact-reward contract 的 diagnostic-only Reference Contact Contract V2 /
完整 21-body active-object pair-force audit。当前 R2 结论为
`REFERENCE_CONTACT_EVIDENCE_INSUFFICIENT`：多个 <=2 cm 的几何强候选手指悬空，
但冻结的 HOCap 输入没有逐指 source-contact 或 topology evidence。V3 保持冻结；
应先补齐或映射该 source evidence，再重跑 Formal20，随后才可授权单独版本化的 V4。

### Causal Decision Tree（FUTURE）

| 观察 | 下一项因果修正 |
| --- | --- |
| Contact 反复断开 | reference-gated contact reward 加 hysteretic contact-loss termination |
| Contact 良好但 terminal dynamics 不稳定 | Contact-ready RSI 加 gravity/friction curriculum |
| 因果表现可接受 | 冻结全局 causal contract，运行第二条 single clip，再进行 Multi-Clip PPO 和 milestone PR 到 `main` |
| 有界 causal corrections 仍不足 | 未来单独创建 `develop/data-H2R` assisted-data branch |

assisted branch 不是 main causal solution。其结果必须标注 external guidance，并声明
`assisted=true`、`causal_physics=false`。

## Milestone

```text
causal single clip -> causal second clip -> Multi-Clip PPO -> milestone PR to main
```

## Milestone 之后

### `feature/ppo-adapter`（FUTURE）

对每个已有 geometry adapter 选择两条代表性数据，执行 PPO、生成 visualization，并用
Evaluation Suite V2 评价。

### `feature/ppo-physical`（FUTURE）

推进更完整的 physical realism：Contact-ready RSI V2、support reconstruction、gravity/friction
curricula、mass/inertia uncertainty、dynamics randomization 和 sensitivity analysis。

## 文档入口

- [Stage 16-D physics contract](stages/STAGE16D_PHYSICS_CONSISTENT_RETARGETING.md)
- [Terminal dynamics attribution](stages/STAGE16D_PHASE1_TERMINAL_DYNAMICS.md)
- [PPO-26D contract](rl/REFERENCE_TRACKING_PPO_26D.md)
- [Evaluation Suite V2](rl/EVALUATION_SUITE_V2.md)
- [Reference Kinematics V2 contract](rl/REFERENCE_KINEMATICS_CONTRACT.md)
- [Phase 3 object-dynamics reward](stages/STAGE16D_PHASE3_OBJECT_DYNAMICS_REWARD.md)
- [Reference-gated contact reward V3](stages/STAGE16D_CONTACT_REWARD_V3.md)
