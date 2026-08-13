# 路线图

本路线图描述可长期复用的科研状态，不是实验流水账。run-specific receipt 保留在忽略的本地存储与详细的
stage/RL 文档中。

## 基础能力

稳定基础包括规范 HOI 数据、语义 MANO 转换、目标手运动学、交互感知重定向、几何/SDF 验证、绑定
manifest 的导出和浏览器证据。Arti-MANO 与 Wuji Hand2 Beta1 仍是支持的目标手路线。

## Stage16-D 因果零重力里程碑（CLOSED）

`CAUSAL_ZERO_G_MILESTONE_COMPLETE`

Stage16-D 冻结一个简化 Isaac/PhysX reference-tracking baseline：

```text
robot action -> hand-object contact -> object dynamics
```

该 baseline 在冻结合同下是物理因果的：gravity 为 zero，support 为 absent，external object guidance
为 absent，且禁止 rollout-time object-state 与 wrist-root write。它不是 physically realistic、
real-world calibrated 或 full-gravity validation。

### 冻结的方法状态

```text
Aggregate V3
  -> STABLE_BASELINE / global default (aggregate_v3)

Strict Per-Finger V4
  -> EXPERIMENTAL_PARTIAL / explicit opt-in (strict_per_finger_v4)
```

V3 是稳定的 reference-gated aggregate fingertip pair-force objective。V4 实现 source-side MANO
contact semantics 与严格独立的 per-finger force credit。它在部分 interaction-fidelity、free-flight
或 twist diagnostic 上改善，但没有在两条 clip 的 physics qualification 中稳定超过 V3；因此它不是
global default。

### 冻结的基础设施

- Reference Kinematics V2
- PPO-26D Isaac Lab backend
- 统一的 V3/V4 contact-reward configuration
- Source Contact Semantics
- Evaluation Suite V2
- full hand-object pair telemetry
- simulation-data export 与 replay diagnostics

历史 V1/V2/V3/V4 artifact 仍通过 provenance-aware compatibility mapping 使用。closeout 不会仅因新
配置默认 V3，就把历史 V4 artifact 静默重解释成 V3。

## Physical Bootstrap P0–P2（COMPLETE；P3 BLOCKED）

`feature/ppo-physical` 已包含 physical bootstrap contract、Contact-ready RSI V2 和
source-support feasibility evidence。P1 使用有界的 full-gravity true-PhysX diagnostic 构造具名
safe reset bank，不引入 guidance、support injection 或 rollout write。P2 在 source support 不可用时绝不
用 generic plane/table 替代。

这不是 PPO 或 curriculum 结果。P3 必须等待当前 absolute hand-object 和 inter-finger geometry gate 通过，
目前保持 blocked。见 [Stage 16 Physical Bootstrap](stages/STAGE16_PHYSICAL_BOOTSTRAP.zh-CN.md)。

## 下一个因果物理阶段

仅在该 milestone PR merge 到 `main` 后，才从 `main` 创建 `feature/ppo-physical`。顺序必须是：

```text
Contact-ready RSI V2
    ↓
Support Feasibility
    ↓
Gravity + Friction Curriculum
    ↓
Full-gravity / zero-guidance qualification
    ↓
Multi-Clip
```

external guidance 或 data-H2R 仍是该 causal physics 路线之后的 assisted fallback，不能替代它。

## 文档入口

- [Stage16-D causal zero-g milestone](stages/STAGE16D_CAUSAL_ZERO_G_MILESTONE.zh-CN.md)
- [Stage 16 Physical Bootstrap](stages/STAGE16_PHYSICAL_BOOTSTRAP.zh-CN.md)
- [Stage 16-D physics contract](stages/STAGE16D_PHYSICS_CONSISTENT_RETARGETING.md)
- [Reference Kinematics V2 contract](rl/REFERENCE_KINEMATICS_CONTRACT.md)
- [PPO-26D contract](rl/REFERENCE_TRACKING_PPO_26D.md)
- [Reference-gated contact reward V3](rl/REFERENCE_GATED_CONTACT_REWARD.zh-CN.md)
- [Strict per-finger contact reward V4](rl/STRICT_PER_FINGER_CONTACT_REWARD.zh-CN.md)
- [Source contact semantics](rl/SOURCE_CONTACT_SEMANTICS.zh-CN.md)
- [Evaluation Suite V2](rl/EVALUATION_SUITE_V2.md)
- [Paper fidelity policy](PAPER_FIDELITY.md)
