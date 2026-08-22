# 路线图

本路线图描述可长期复用的科研状态，不是实验流水账。run-specific receipt 保留在忽略的本地存储与详细的
stage/RL 文档中。

## 基础能力

稳定基础包括规范 HOI 数据、语义 MANO 转换、目标手运动学、交互感知重定向、几何/SDF 验证、绑定
manifest 的导出和浏览器证据。Arti-MANO 与 Wuji Hand2 Beta1 仍是支持的目标手路线。

## Stage16 原始 mocap replay overlay（已实现）

authoritative IsaacLab replay 现在可在同一 world frame/timeline 中显示记录的 PhysX actual、按
provenance 解析的原始 HOCap MANO/object ghost，以及可区分的 geometric-retarget reference ghost。
raw layer 仅用于可视化；coordinate/time alignment 是 deterministic 且 fail-closed。它支持
object-local fingertip diagnostic，但不修改 PPO、reward、physics、controller 或 reference。见
[raw-mocap replay overlay](rl/RAW_MOCAP_REPLAY_OVERLAY.md)。

## Stage16 contact timing、angular twist、PF 与 DF（已实现）

主线证据顺序现在固定为：

```text
Raw Mocap
    -> Geometric Retarget
    -> Physical Functionality
     + Demonstration Fidelity
```

`SR_dynamic` receipt 保持 immutable。离线合同将 raw-MANO、retarget-reference 与 PhysX
contact timing 分层，比较 trace omega 与 pose-derived omega，并将 physical completion（PF）
与 demonstration fidelity（DF）分开。

angular semantics closeout 确认历史 trace 字段是 world-frame、instantaneous PhysX COM
angular velocity，并采用 `Stage16ActualAngularVelocityAuthorityV2`：从保存的 actual pose
使用与 Reference Kinematics V2 相同的 control-rate SO(3)-log estimator 计算 actual omega。
V4/170650 从 legacy trace 字段下的 2/20 变为 comparable semantics 下的 20/20；没有重写
trace，也没有调整 inherited threshold。

raw grasp review 证明 Strict V4 是 reward-specific 的 named-finger 到 robot-tip target，
不是经验证的 functional human-grasp binary。新增的
`RawHumanGraspReadinessProfileV1` 分别报告 all-surface、multi-region、topology 与 coupling
层。170105 的 any-surface contact 略早于 LIFT，但 multi-region 与 Strict-V4 readiness
均晚于 LIFT；functional raw readiness 仍为 `NOT_IDENTIFIABLE`。因此 contact-timing
attribution 仍为 `INCONCLUSIVE`，现为 medium confidence 且显式采用 profile-based 解释。
见 [actual angular velocity semantics](rl/ACTUAL_ANGULAR_VELOCITY_SEMANTICS.md)、
[raw human grasp readiness authority](rl/RAW_HUMAN_GRASP_READINESS_AUTHORITY.md)、
[contact timing attribution](rl/CONTACT_TIMING_LAYER_ATTRIBUTION.md)、
[angular-twist audit](rl/ANGULAR_TWIST_AUDIT.md) 与
[PF/DF](rl/PHYSICAL_FUNCTIONALITY_AND_DEMONSTRATION_FIDELITY.md)。

## Stage16 PF V2 因果 lift 与对称 PPO（已收口）

`Stage16PhysicalFunctionalityV2` 是冻结的附加因果 lift authority，不重写 PF V1。其 support
proxy 取自独立记录的 table ContactSensor reset sample，而非 replay 中的桌面几何或手-物
pair-force validity mask。修正后的历史 170650 positive control 为 PF V2 20/20。170105 从
U10 接续到 U11 的 Confirm20 取得 PF V2 与三个 DF 维度 20/20；历史 PF V1 的 timing gate
保持不变，故仍为 0/20。新的 170650 experimental continuation 首次在 U2 达到最大值（20/20），
之后并非单调：U8 为 0/10，U10 Eval10 恢复为 10/10；历史 accepted actor 仍被冻结。唯一后续是
无 tuning 地诊断 170650 continuation instability。见 [PF V2 因果 lift 审计](rl/PHYSICAL_FUNCTIONALITY_V2_CAUSAL_LIFT.md)。

## Stage16 170650 physical-HOI 收口与通用 profile（CLOSED）

V4/`hocap_170650` 已正式成为 `ACCEPTED_STAGE16_PHYSICAL_HOI`：PF、pose、linear、
Angular-Authority-V2、causality 与 geometry 均为 20/20。该 lineage 现在冻结为
physical-HOI 数据源/positive control，不再需要 PPO 或 policy adaptation。

`HumanObjectCouplingContactProfileV1` 已为 `hocap_170105` 与 `hocap_170650` 描述 raw
contact geometry、regions、geometric topology、`T_H^-1 T_O`、relative motion 与 continuous
coupling。由于 immutable PhysX trace 缺少 actual contact points/normals 与 exact slip，其跨层状态为
`PROFILE_PARTIALLY_VALIDATED`。下一步唯一 generic refinement family 是 medium-confidence 的
`SOURCE_PROFILE_TRACKING`；support transfer 只作为 outcome metric。本任务不实施 refinement，也不启动训练。
见 [170650 acceptance](rl/STAGE16_170650_ACCEPTANCE.md)、[coupling contact
profile](rl/HUMAN_OBJECT_COUPLING_CONTACT_PROFILE.md) 与 [generic refinement
target](rl/GENERIC_PHYSICAL_REFINEMENT_TARGET.md)。

## Stage16 SourceProfileTracking V1 离线硬门（CLOSED，未提升）

第一次 object-agnostic V1 实现复用了冻结的人-物 profile，并采用全局归一化的 activity、object-local
geometry 与 pose-derived coupling residual。目标数值有限，accepted 170650 positive control 也没有异常；但
CONTACT-to-early-LIFT 的必需排序失败：170105 C4 failure 的 combined profile loss 反而更低。最终为
`OFFLINE_OBJECTIVE_INVALID` / `PROFILE_OBJECTIVE_NOT_DISCRIMINATIVE`；没有加入 profile reward、没有 PPO
update，也没有训练 170650。见 [SourceProfileTracking V1](rl/SOURCE_PROFILE_TRACKING_REFINEMENT.md)。

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

## Causal Physical PPO 重力 curriculum（EXECUTION）

`feature/ppo-physical` 已包含 physical bootstrap contract、Contact-ready RSI V2、
source-support feasibility evidence 与 staged gravity/friction curriculum。P1 使用有界的
full-gravity true-PhysX diagnostic 构造具名 safe reset bank，不引入 guidance、support injection
或 rollout write。P2 在 source support 不可用时绝不用 generic plane/table 替代。

explicit virtual wrist 的 C4 rotational controller repair 已完成：PhysX 在
reduced-coordinate articulation import 后没有让生成 USD 的 per-body hand-gravity opinion 生效，
所以 production spawn 现在应用等价的 runtime articulation override。task object 的 gravity 仍为 ON，
也不授权修改 PPO、reward、action 或 physical qualification。见
[wrist controller root cause](rl/WRIST_ROTATIONAL_CONTROLLER_ROOT_CAUSE.md)。

当前执行路线将 V3/V4 × 两个 clip 的四条冻结 lineage 从 zero-g 连续运行至 C0--C4。每个固定 sample
budget 完成即进入下一 stage。saturation、optimization health、interaction、twist、penetration、reference
geometry 与 Evaluation Suite V2 都是最终诊断，绝不再作为 PPO curriculum stop gate。见
[Physics curriculum](rl/PHYSICS_CURRICULUM.md)。

冻结 V3 `hocap_170105` C1 saturation gate 现有 durable pre-gate instrumentation；它只用于诊断和
可复现性，不授权 C2，也不改变 physical route。见 [C1 saturation
instrumentation](rl/C1_SATURATION_INSTRUMENTATION.md)。

其历史重跑仍仅作为 attribution 证据。保留的 0.98/0.25 saturation threshold 现在只输出 telemetry warning，
不会阻止 C1--C4 continuation。
有界的 P3-C1.2 PPO optimization attribution 记录在 [C1 PPO optimization
attribution](rl/C1_PPO_OPTIMIZATION_ATTRIBUTION.md)。若缺少 exact PPO batch
证据而得到 `INCONCLUSIVE`，仍保持 fail-closed，不授权 formal P3 继续。

C0 contact-skill-collapse 审计将第一次瞬时 contact 丢失定位到 PPO update 3 / 122,880
samples，并确认 frame0-only training reset 是主要原因。冻结的 uniform-RSI `[0,320]`
反事实在 update 6 之前始终保留 10/10 deterministic frame-0 contact 与 lift，且 reward、
controller、reference、action 和 runtime-write contract 均未改变。因此 C0 physical
training 默认恢复 uniform RSI；formal evaluation 仍固定 frame0。受限的
V3/hocap_170105 continuation 已从精确 U6 state 完成 C0 和 C1：两个 endpoint 均保留
10/10 frame0 contact，但 C0/C1 endpoint 的 lift 均为 0/10，C1 physics 为
0.25g / 1.75x friction。这不授权 C2--C4 或 four-lineage rerun。见
[contact-skill collapse localization](rl/CONTACT_SKILL_COLLAPSE.md)。

后续的离线 grasp/lift localization 使用已保存的 46 个新 checkpoint/exact batch
以及 480 条 frame0 trace，未重新训练。它将 C0 grasp/lift 转折精确定位为
U25 -> U26：U25 为 10/10 persistent grasp 与 lift，U26 则为 0/10，只剩晚到的
grazing contact。冻结的 U26 APPROACH/reference-contact/GRASP restart 均仍为
0/10 lift，而 U25 GRASP 对照为 10/10。因此保持 fail-closed 结论
`PPO_OPTIMIZATION_FORGETTING_PRIMARY`，而不是 reward shortcut 或 controller
regression。唯一允许的下一步是
`NEXT_CONTACT_SKILL_POLICY_PRESERVATION_ABLATION`；不得提升 C0 endpoint，
不得启动 C2--C4。见 [contact-skill collapse
localization](rl/CONTACT_SKILL_COLLAPSE.md)。

exact-batch policy-preservation ablation 在 paired actor-only/critic-baseline
shadow replay 后选择了 opt-in 的 0.50x actor-LR candidate。该 single-update 结果
保留 10/10 frame0 grasp/lift 与 U25 GRASP reset，但并不能证明 live training 中的
长期保留。已完成的 V3/`hocap_170105` full-C0 validation 覆盖 26 个 update、
1,048,576 samples；候选在 U17 / 696,320 samples 即同时失去 grasp 与 lift，之后
均为 0/10，endpoint Eval20 为 0/20。相比之下，冻结的 1.0x C0 lineage 在 U25
仍为 10/10，首次 collapse 在 U26。实际 classification 为
`CANDIDATE_REGRESSION`、`STATUS=SHADOW_ONLY_NOT_SUFFICIENT`。不得切换 production
default，不得启动 C1；唯一允许的后续动作是
`NEXT_UPDATE_DEPTH_POLICY_PRESERVATION_ABLATION`。见 [full C0 longitudinal
validation](rl/CONTACT_PRESERVING_FULL_C0_VALIDATION.md)。

这仍是历史 C0 optimization-preservation 结论，不再是当前 physical program 的主动作。
冻结 source 的 C0--C4 gravity/friction sweep 已完成 isolated-process 的 timeout/terminal-capture
修复和授权的最小 adaptation 决策树。四条 C4 receipt 均已技术完成。历史 V4/170650
`SR_dynamic V1` 结果为 2/20；后续采用 comparable semantics 的 Authority-V2 requalification
已以 PF/DF 20/20 取代它作为 physical-HOI acceptance authority。V3/170650 仅恢复到 C2，完整
C3 budget 失败；V3/170105 的 C1 和 V4/170105 的 C4 均耗尽 budget，未恢复 lift。不得继续
PPO/reward/LR sweep；accepted V4/170650 保持冻结，170105 只进入未来选定的 object-agnostic
profile-tracking refinement。见
[full-gravity capability closure](rl/FULL_GRAVITY_CAPABILITY_CLOSURE.md)。

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

- [Dexplore 风格乘法 reward 与 RSE](rl/DEXPLORE_STYLE_MULTIPLICATIVE_REWARD_RSE.md)
- [Stage16-D causal zero-g milestone](stages/STAGE16D_CAUSAL_ZERO_G_MILESTONE.zh-CN.md)
- [Stage 16 Physical Bootstrap](stages/STAGE16_PHYSICAL_BOOTSTRAP.zh-CN.md)
- [Physics curriculum](rl/PHYSICS_CURRICULUM.md)
- [Hand gravity control abstraction](rl/HAND_GRAVITY_CONTROL_ABSTRACTION.md)
- [Wrist rotational controller root cause](rl/WRIST_ROTATIONAL_CONTROLLER_ROOT_CAUSE.md)
- [Stage16 full-gravity causal status](stages/STAGE16_FULL_GRAVITY_CAUSAL.md)
- [Stage 16-D physics contract](stages/STAGE16D_PHYSICS_CONSISTENT_RETARGETING.md)
- [Reference Kinematics V2 contract](rl/REFERENCE_KINEMATICS_CONTRACT.md)
- [PPO-26D contract](rl/REFERENCE_TRACKING_PPO_26D.md)
- [C1 PPO optimization attribution](rl/C1_PPO_OPTIMIZATION_ATTRIBUTION.md)
- [Contact-preserving full C0 validation](rl/CONTACT_PRESERVING_FULL_C0_VALIDATION.md)
- [Frozen source policy gravity sweep](rl/FROZEN_SOURCE_POLICY_GRAVITY_SWEEP.md)
- [Actual angular velocity semantics](rl/ACTUAL_ANGULAR_VELOCITY_SEMANTICS.md)
- [Raw human grasp readiness authority](rl/RAW_HUMAN_GRASP_READINESS_AUTHORITY.md)
- [Reference-gated contact reward V3](rl/REFERENCE_GATED_CONTACT_REWARD.zh-CN.md)
- [Strict per-finger contact reward V4](rl/STRICT_PER_FINGER_CONTACT_REWARD.zh-CN.md)
- [Source contact semantics](rl/SOURCE_CONTACT_SEMANTICS.zh-CN.md)
- [Evaluation Suite V2](rl/EVALUATION_SUITE_V2.md)
- [Paper fidelity policy](PAPER_FIDELITY.md)
## P3-B.7 重启合同

P3-B.7 严格区分不可修改的几何参考（诊断性的 soft target）与物理合法的
hard reset、实际 PPO 轨迹（两者均为 hard gate）。整条参考轨迹的几何失败
本身不再阻止训练；缺少安全的早期桌面支撑 reset 才会阻止重启。

## Stage16 grouped-reward/RSE 有界 refinement

opt-in 的 grouped multiplicative reward 与 reference-scoped exploration 已通过
offline 与 no-step runtime gate，随后完成原始预注册的十次
V4/`hocap_170105`/C4 update（409,600 samples）。U10 将 legacy lift 提升到 6/10，
但 persistent multi-contact 晚于 LIFT，故 PF V1 仍为 0/10。修正 PF V2 support-sensor 审计后，
U10 又执行一次有界 update：U11 的 PF V2 Eval10 为 10/10，Confirm20 的 PF V2 与三个 DF
维度均为 20/20，PF V1 则保持 0/20。独立的 historical-170650 experimental continuation
完成自己的 U1--U10 预算；其首次最大值为 U2 Eval20 20/20，之后 Eval10 非单调（U8 为 0/10，
U10 恢复到 10/10）。冻结的 accepted 170650 actor 未被改写。见
[PF V2 causal-lift audit](rl/PHYSICAL_FUNCTIONALITY_V2_CAUSAL_LIFT.md)。
