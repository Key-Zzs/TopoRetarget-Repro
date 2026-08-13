# Stage16-D 因果零重力里程碑

## 范围

Stage16-D 已作为冻结、简化 Isaac/PhysX 合同下的物理因果 reference-tracking baseline 收口：

```text
robot action -> hand-object contact -> object dynamics
```

合同使用 zero gravity、no support、no external object guidance、no rollout-time object-state write
和 no rollout-time wrist-root write。它是因果仿真 baseline，不是 physically realistic、real-world
calibrated 或 full-gravity validation。

## 稳定基线

**Aggregate V3** 是 `STABLE_BASELINE`，并通过以下配置成为 global default：

```yaml
reward:
  contact:
    mode: aggregate_v3
```

它是冻结的 reference-gated aggregate fingertip pair-force objective，也是下一物理阶段的 baseline。

## 实验目标

**Strict Per-Finger V4** 是 `EXPERIMENTAL_PARTIAL`，必须明确 opt-in：

```yaml
reward:
  contact:
    mode: strict_per_finger_v4
```

它使用 source-side MANO/object contact semantics，只给匹配 named fingertip 的 active-object PhysX
pair force 记分。V4 已完整实现，但没有在两条 clip 的 physics qualification 中稳定优于 V3，因此不是
global default。

## 保留的基础设施

- Reference Kinematics V2
- 26-D PPO residual action 与 Isaac Lab backend
- 统一的 contact-reward mode / legacy provenance mapping
- Source Contact Semantics 与完整 21-body pair-force telemetry
- Evaluation Suite V2
- replay diagnostics 与 simulation-data export

历史 V1/V2/V3/V4 checkpoint、trace 和导出的 simulation data 保留其记录的 provenance。缺失的 legacy
mode field 只会由无歧义的 V3/V4 reward-contract identifier 映射；没有 hidden fallback。

## 已知限制

- no gravity
- no support
- 冻结的简化 damping 与 contact 假设
- 无 real-world calibration
- V4 并非稳定优于 V3

## 下一步

PR merge 到 `main` 后，下一个 branch 才可按以下顺序启动 physical curriculum：Contact-ready RSI V2、
Support Feasibility、Gravity + Friction Curriculum、Full-gravity / zero-guidance qualification，最后才是
Multi-Clip。external guidance/data-H2R 仍是 causal physics 路线之后的 assisted fallback。

## Fidelity 标签

factor-eight retiming、26-D world-wrist extension、V3/V4 objective 与 Isaac Lab backend 都是 engineering
extension。它们受仓库的 paper-fidelity policy 约束，不能标成 author-exact。
