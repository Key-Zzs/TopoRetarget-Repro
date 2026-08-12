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

### Reference-Gated Contact Reward V3（FROZEN BASELINE）

V3 是历史 aggregate baseline，使用 reference-only 的 3 cm Wuji distal-root proximity mask 和当前
named-tip PhysX pair-force magnitude 的 aggregate sum。它为比较而保留，后续 source-contact
semantics 不会改动它。

### Source Contact Semantics（VALIDATED）

`SourcePerFingerContactEvidenceV1` 建立 raw HOCap MANO/object surface 的逐 finger contact，及其到
321 runtime frame 的冻结 factor-eight mapping。它区分 confirmed、persistent-confirmed、probable、
transition、proximity-only、no-contact 和 ambiguous evidence。最终 audit 以高置信度将 Strict
Per-Finger V4 选为 V3 的唯一 successor。

### Strict Per-Finger Contact Reward V4（CURRENT）

V4 替换 V3 aggregate contact term，同时保留冻结的 Reward V2 components、Reference Kinematics V2、
764-D observation、26-D action、physics、controller 和 PPO hyperparameters。source-confirmed 或
persistent-confirmed 的 finger `f` requirement 只可由 finger `f` 的 named distal/tip-to-active-object
pair force 获得 contact reward。contact term 按 source-required finger 数量归一化；不得使用
whole-hand force、same-finger group force 或 cross-finger compensation。

V4 因果路线为：

```text
Source Contact Semantics
    VALIDATED
        ↓
Strict Per-Finger Reward V4
    CURRENT
        ↓
if validated:
    Freeze Causal Contact Reward
        ↓
    Contact-ready RSI V2
        ↓
    Support Feasibility
        ↓
    Gravity + Friction Curriculum
        ↓
    Full-gravity / zero-guidance Formal Qualification
        ↓
    Multi-Clip
        ↓
    causal milestone
        ↓
only if causal path insufficient:
    external guidance / data-H2R
```

V4 不加入 object guidance、object-state write、attachment、suction、contact-loss termination、terminal
reward、penetration reward、gravity/friction curriculum、Multi-Clip PPO 或 data-H2R。gravity/physics
curriculum 明确位于任何 external-guidance route 之前。

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
- [Strict per-finger contact reward V4](rl/STRICT_PER_FINGER_CONTACT_REWARD.zh-CN.md)
- [Stage 16-D Strict Per-Finger V4](stages/STAGE16D_STRICT_PER_FINGER_V4.zh-CN.md)
