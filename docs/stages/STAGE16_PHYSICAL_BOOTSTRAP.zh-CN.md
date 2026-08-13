# Stage 16 物理 Bootstrap（P0–P2）

本文定义从已关闭的 Stage16-D 因果零重力 baseline 到可能的物理 PPO 阶段之间的桥接。它不是 PPO、
gravity curriculum 或 friction curriculum 的结果。

机器可读合同：

- `configs/rl/stage16/stage16_physical_bootstrap.yaml`
- `configs/rl/stage16/stage16_contact_ready_rsi_v2.yaml`
- `configs/rl/stage16/stage16_p3_entry_gate_v1.yaml`

物化命令为 `scripts/evaluation/run_stage16_physical_p0_p2.py`。其报告是位于
`.local/reports/stage16_physical_p0_p2` 的本地、忽略的 diagnostic evidence。

## P0：冻结的物理边界

P0 对 Stage16-D parent、Reference Kinematics V2、PPO-26D action interface、当前 nominal
asset/contact configuration、Source Contact Semantics 与 Evaluation Suite V2 做 hash 可追溯绑定。
目标 gravity 是 `(0, 0, -9.81) m/s²`，标记为 `EARTH_NOMINAL_ENGINEERING_TARGET`；它不是 source
calibration 的声明。

合同禁止 external guidance、rollout object-state write 与 rollout wrist-root write。object 与 wrist
state write 只允许在 reset 时发生。未知 mode string 或损坏的 provenance 都必须 fail closed。

## P1：Contact-ready RSI V2

RSI V2 从三类证据为每个 V2 runtime index 分类：

1. Source Contact Semantics 的 confirmed/persistent label 是 contact truth。
2. retargeted Wuji link-to-object-axis distance 仅记录为 geometry evidence，不能凭空制造 contact。
3. V2 object twist 用于区分 manipulation 和 terminal hold。

类别为 `PRE_CONTACT`、`NEAR_CONTACT`、`CONTACT_READY`、`PERSISTENT_CONTACT`、`MANIPULATION`、
`TERMINAL_HOLD` 与 `AMBIGUOUS`。`PRE_CONTACT` 和 `AMBIGUOUS` 是显式 invalid reset state。旧的 V3
three-centimetre reward mask 被禁止作为 RSI truth。

有界 gravity diagnostic 使用真实 Isaac/PhysX、nominal gravity、zero policy residual 加 reference
following、每 state 四个 replica 和 20 个 control step（一秒）。它记录 contact timing/persistence、
pre-contact displacement/velocity、joint/finite/catastrophic outcome 与 rollout-write counter；不训练 PPO，
也不添加 support mesh。

只有同一 state 的每个 replica 都通过预先声明的 engineering threshold，才可进入 safe bank。初始 P3
只允许 `CONTACT_READY_SAFE`、`PERSISTENT_SAFE` 和 `MANIPULATION_SAFE`；`NEAR_CONTACT_SAFE` 与
`TERMINAL_SAFE` 默认仅用于诊断。

## P2：support feasibility

support 使用以下证据层级：

1. 显式 source scene/support asset；
2. 带 provenance 的可恢复 source scene geometry；
3. 真实 PhysX P1 diagnostic 的 hand-support evidence；
4. 否则为 `SUPPORT_UNKNOWN`。

infinite ground plane、generic table、fixture、attachment 或 hidden support 都绝不能成为自动 fallback。特别地，
没有可恢复 support asset 的 source 不能授权 frame-zero full-gravity reset。

若 P1 提供 contact-ready safe bank 而 source support 不可用，P2 可以只授权
`CONTACT_READY_ONLY_VALIDATED`。这是受约束的 reset policy，并不证明 source object 在 frame zero
由 hand 支撑。

## P3 决策合同

P3 只是 entry decision，绝不启动 trainer。冻结 gate 为：

- G0 provenance 和 parent contract；
- G1 Contact-ready RSI V2；
- G2 source-support 或受约束 contact-ready feasibility；
- G3 当前 absolute hand-object 与 inter-finger geometry gate；
- G4 针对授权 reset bank 的 controller/actuator 与 joint-limit safety；
- G5 zero guidance、no hidden support 与 no prohibited rollout write。

历史 zero-gravity geometry artifact 绝不能被重标为 full-gravity G3 pass。当前物理状态若未执行 G3，P3
就是 `P3_BLOCKED_TECHNICAL`，即使 P0/P1/P2 已通过。所有 gate 真实通过后的
`P3_READY_WITH_CONSTRAINTS` 仍只允许具名 safe bank，并且没有 source-backed support 时绝不允许
frame-zero full-gravity reproduction。

## Reproduction 边界

只在输入报告已存在时运行 P0/P1/P2：

```bash
conda run -n toporetarget-rl python scripts/evaluation/run_stage16_physical_p0_p2.py p0
conda run -n toporetarget-rl python scripts/evaluation/run_stage16_physical_p0_p2.py build-p1-banks
# 运行有界 fresh Isaac worker，再合并其 COMPLETE receipt。
conda run -n toporetarget-rl python scripts/evaluation/run_stage16_physical_p0_p2.py run-p1-diagnostics
conda run -n toporetarget-rl python scripts/evaluation/run_stage16_physical_p0_p2.py merge-p1-diagnostics
conda run -n toporetarget-rl python scripts/evaluation/run_stage16_physical_p0_p2.py finalize-p1
conda run -n toporetarget-rl python scripts/evaluation/run_stage16_physical_p0_p2.py finalize-p2
conda run -n toporetarget-rl python scripts/evaluation/run_stage16_physical_p0_p2.py finalize
```

首次 P3 PPO run、任何 gravity curriculum、任何 friction curriculum 以及任何 P4 human decision 都需要
单独授权。
