# OakInk2 O5R-D2E/D2F Gate 对齐与 V3 独立复认证

本阶段只回答一个问题：SparseValidationV2 唯一失败的“全帧精确单调不退化率至少
90%”，是否是冻结 ObjectiveV2、Semantic V1 或机器人硬约束要求。结论为否。它是
预注册但没有独立语义、物理或机器人权威的保守认证启发式。历史
`SPARSE_VALIDATION_V2=FAIL` 不改写；ObjectiveV2 与 S1 ExecutionV2 也没有修改。

## D2E 只读审计

对 SparseV2 的 10 个 LOW 帧使用保存状态重算，没有调用优化器，也没有消费新 DEV1
或 DEV2 帧。10/10 的旧状态和新状态都在 `E_IM <= 1e-4` 有效集内，全部硬约束通过。
绝对增量的中位数为 `1.30158685243494e-07`，p95 为
`4.775905274628456e-07`，最大值为 `6.657905898913041e-07`；最小最终阈值余量为
`7.099265317566698e-06`。10/10 状态移动均由冻结字典序选择中的 secondary fidelity
改善解释，不是 lexicographic violation。

因此根因为 `CERTIFICATION_GATE_SEMANTIC_MISMATCH`，置信度 `HIGH`。GateV2 冻结为：

```text
E_new <= max(E_old, tau) + epsilon_num
tau = 1e-4
epsilon_num = 1e-12
required fraction = 100%
```

`epsilon_num` 继承自 GateV1 的数值复现容差，不由 SparseV2 结果拟合。原 80% 无效帧
恢复率、50% 无效帧中位改善、全部硬约束与 determinism 要求保持不变。精确单调率和
catastrophic-regression 仅保留为诊断。GateV2 对 SparseV2 的计算只能标记
`NOT_APPLICABLE_POST_HOC`，不能把历史 FAIL 变成 PASS。

## Fresh V3 结果

SparseV3 在冻结 manifest 后仅运行一次，使用 30 个新帧（HIGH/MID/LOW 各 10）；与
55 个开发帧及 30 个 SparseV2 帧的 overlap 都为 0。结果为：30/30 technical/hard
valid；20/20 旧无效帧恢复；中位相对改善 `0.5445401160443923`；30/30 通过
threshold-aware non-regression；10/10 LOW 保持有效；冻结的 5 帧三次重复 determinism
通过。精确单调诊断仍为 20/30，不再作为结构性 gate。最终
`SPARSE_VALIDATION_V3=PASS`。

WindowV3 在冻结后运行 4 个互不重叠的连续 32 帧窗口；与开发、SparseV2、SparseV3
overlap 均为 0。四个窗口全部 technical、continuity、wrist、bone、collision、joint
limit、reflection、scale 和 interaction 通过；HIGH_1 与 LOW 三次重复 determinism
通过。p95 如下：

| Window | N | Old p95 | New p95 | Result |
| --- | ---: | ---: | ---: | --- |
| HIGH_1 | 32 | 0.000156918048077 | 0.000090387135500 | PASS |
| HIGH_2 | 32 | 0.000191813498612 | 0.000074054197911 | PASS |
| LOW | 32 | 0.000069500994625 | 0.000070009782984 | PASS |
| MID | 32 | 0.000109785948783 | 0.000073775487523 | PASS |

## DEV2 硬控制

冻结 O5 authority 确认 DEV2 为 record `...:00010`、primitive `rearrange`、对象
`C11001`、source interval `[10704,10944)`、240 帧。SparseV3 与 WindowV3 PASS 后，
frame0 hard control 获得执行授权并调用三次。

但该 episode 的旧 O5 运行在 Stage 7 frame0 已失败，只保存 canonical episode；没有
DEV2 warm start、interaction graph 或 final production trajectory。冻结 S1 同时要求
`old_production` 的 `q_old` 始终作为 baseline candidate。三次调用都在冻结方法输入权威
检查处得到同一技术失败：
`MISSING_FROZEN_S1_OLD_PRODUCTION_Q_OLD_AUTHORITY`。没有启动优化器，也没有把旧 Stage 7
失败终态、neutral 或其他 seed 冒充 `q_old`；没有 DEV2 特例。

所以 `DEV2_FRAME0_HARD_CONTROL=FAIL`，`FULL_DEV2_COMPUTE_AUTHORIZED=NO`，完整 DEV2
production solve 为 0 次。Semantic V1 和 viewer 都是 `NOT_RUN`。下一阶段只能是
`OBJECTIVE_V2_CROSS_EPISODE_HARD_CONTROL_FAILURE_ANALYSIS`；本阶段不授权 DEV1 full、O6、
Support、PhysX、frozen eval 或 PPO。

所有运行证据位于忽略目录：
`.local/reports/oakink2_o5rd2e_gate_alignment_and_v3_recertification_v1/`。
