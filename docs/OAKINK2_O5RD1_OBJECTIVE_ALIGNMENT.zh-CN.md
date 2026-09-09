# OakInk2 O5R-D1 objective alignment 审计

O5R-D1 是诊断审计，不是新的优化方法。它只复用 25 个 B1 development frames
和 30 个已经失败、因此现已被消费的 O5R-C SparseValidationV1 frames，解释
O5R-B 找到的低 interaction-error 状态为什么没有成为 production solution。

## 结论

`StructuredSolverV1` 已被独立验证科学拒绝，不得再作为 production solver
复用。它虽然 30/30 技术完成，但未通过冻结的 Semantic V1 sparse gate。

Production objective 确实精确包含 Semantic V1 的逐帧 graph-Laplacian
`E_IM`，权重为 500；但这不等于完整 objective 与语义门对齐。逐帧标量目标
还包含 bone fidelity、continuous temporal regularization、base correction 和
collision slack，而且没有直接优化轨迹级 Semantic V1 p95 gate。

25/25 个 B1 frames 中，Q1 的 `E_IM` 更低，但精确 production objective
全部更差。Q0 到 Q1 的加权项变化中位数为：

- continuous temporal：`+9.961803871`
- bone direction：`+0.075531161`
- interaction mesh：`-0.040541888`

25 个 Q1 状态全部满足 production hard-collision bound，所以 Q1 未被保留
不是硬可行性问题，而是 objective trade-off。10 条确定性插值路径均表现为
语义误差单调改善、production objective 单调恶化。另一个冻结的 10-frame
Q1-start micro-polish control 中，full polish 在 10/10 帧上拉坏 `E_IM`，同时
在 10/10 帧上改善精确 production objective。

Contributor selection 不是主因：dominant contributor 与 selected block 在
55/55 帧上一致。Structured block stage 很弱且不稳定（32 effective、22
regressed、1 numerically weak）。常规 Q2 到 Q3 在该总体上并不是 rollback；
它通常继续改善 `E_IM`。

因此主要根因是 `OBJECTIVE_SEMANTIC_MISALIGNMENT`，置信度 `HIGH`。此外，
O5R-C 在构造 structured-stage context 时没有传入 production continuous
prediction inputs，导致运行了另一条 temporal branch。这是 V1 的次要 binding
缺陷，但不会恢复已经被拒绝的 solver 的科学有效性。

## 复现

真实 CLI 为 `scripts/data/run_oakink2_o5rd1.py`。先运行：

```bash
conda run -n toporetarget-rl \
  python scripts/data/run_oakink2_o5rd1.py --help
```

各 action 可分别执行 objective audit、Q2/Q3 replay、Q0-Q3 decomposition、
contributor/block 与 polish rollback 审计，以及有界 directional、interpolation
和 development-only micro-polish 诊断。证据统一写入
`.local/reports/oakink2_o5rd1_objective_alignment_v1/`。

## 未来验证卫生

25 个 B1 frames 与 30 个 SparseValidationV1 frames 已永久作为方法开发/诊断
样本消费。未来 V2 必须先冻结新方法 contract，再从未消费帧中冻结新的
SparseValidationV2，并排除这 55 个 frame IDs。O5R-D1 没有创建 V2 objective、
solver、semantic metric 或 validation set，也没有运行任何 DEV1 full trajectory
或 DEV2 solve。
