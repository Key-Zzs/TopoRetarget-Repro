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

## Physical route P0–P3（P3 在 C2 selection 处 BLOCKED）

`feature/ppo-physical` 已包含 physical bootstrap contract、Contact-ready RSI V2、
source-support feasibility evidence 与 staged gravity/friction curriculum。P1 使用有界的
full-gravity true-PhysX diagnostic 构造具名 safe reset bank，不引入 guidance、support injection
或 rollout write。P2 在 source support 不可用时绝不用 generic plane/table 替代。

P3 C0--C2 development pilot 已对两个冻结 reward mode 和两个 clip 完成。必需的 global C2 selection
因两个 mode 都未通过 absolute geometry gate 而拒绝它们。因此在 G3 与 C3/C4 前 fail-closed；P4 未运行，
不存在 full-gravity causal result。见 [Physics curriculum](rl/PHYSICS_CURRICULUM.md) 与
[Stage16 full-gravity causal status](stages/STAGE16_FULL_GRAVITY_CAUSAL.md)。

### Support resolution reconstruction（已实现，不提升）

source-first resolver、stable-pre-contact 平面推断、有限运行时 proxy、几何 audit 以及 object-only 全重力 PhysX A/B
已经对 `hocap_170105` 与 `hocap_170650` 实现。两段都没有可恢复的 source support，因此使用显式标注的 inferred plane。
proxy 持续产生接触且法向力约为 `mg`，nominal object-only 的位置与四元数姿态保持稳定。因此支撑 contract 已通过，
但 runtime transfer 受现有 hand-object geometry blocker 延后；这不授权向 RL environment 加桌面，也不推进 P3/G3/P4。
见 [Support resolution](physics/SUPPORT_RESOLUTION.zh-CN.md)。

## 下一个因果物理阶段

P3-B.5 已确认 C2 的主要原因是 reset geometry：所选 safe-bank state 在所有冻结的
A/B/C/D 反事实中第 0 帧即违反 geometry gate。接下来允许的工作是在冻结 causal contract 下
显式修复 C2 absolute-geometry failure。修复后的顺序必须是：

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

P3-B.6 已完成两个 clip 的 321 帧 physical mask、finite-support RSI bank、dynamic
reset 与 joint zero-replay receipt。两个 clip 仍为
`P3_RESTART_BLOCKED_REFERENCE_GEOMETRY`，没有启动 PPO。见 [P3-B.6 物理场景与 RSI
再资格化](rl/PHYSICAL_SCENE_RSI_REQUALIFICATION.zh-CN.md)。

external guidance 或 data-H2R 仍是该 causal physics 路线之后的 assisted fallback，不能替代它。

## 文档入口

- [Stage16-D causal zero-g milestone](stages/STAGE16D_CAUSAL_ZERO_G_MILESTONE.zh-CN.md)
- [Stage 16 Physical Bootstrap](stages/STAGE16_PHYSICAL_BOOTSTRAP.zh-CN.md)
- [Physics curriculum](rl/PHYSICS_CURRICULUM.md)
- [Stage16 full-gravity causal status](stages/STAGE16_FULL_GRAVITY_CAUSAL.md)
- [Stage 16-D physics contract](stages/STAGE16D_PHYSICS_CONSISTENT_RETARGETING.md)
- [Reference Kinematics V2 contract](rl/REFERENCE_KINEMATICS_CONTRACT.md)
- [PPO-26D contract](rl/REFERENCE_TRACKING_PPO_26D.md)
- [Reference-gated contact reward V3](rl/REFERENCE_GATED_CONTACT_REWARD.zh-CN.md)
- [Strict per-finger contact reward V4](rl/STRICT_PER_FINGER_CONTACT_REWARD.zh-CN.md)
- [Source contact semantics](rl/SOURCE_CONTACT_SEMANTICS.zh-CN.md)
- [Evaluation Suite V2](rl/EVALUATION_SUITE_V2.md)
- [Paper fidelity policy](PAPER_FIDELITY.md)
## P3-B.7 重启合同

P3-B.7 严格区分不可修改的几何参考（诊断性的 soft target）与物理合法的
hard reset、实际 PPO 轨迹（两者均为 hard gate）。整条参考轨迹的几何失败
本身不再阻止训练；缺少安全的早期桌面支撑 reset 才会阻止重启。
