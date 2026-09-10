# OakInk2 O5R-D2D：ObjectiveV2 独立认证

O5R-D2D 对 O5R-D2C 冻结的 Candidate B2 与 `S1_DETERMINISTIC_MULTI_START` 做 method-development-independent、DEV1 episode 内的帧级/窗口级认证。它不是新物体、新 episode 或最终 corpus heldout 泛化测试。

冻结 authority：

- `RetargetObjectiveV2 SHA256=48861d02020395bbdd891635e2176028f3fd5a86f825c784a6763dc3504211bc`
- `ExecutionV2 SHA256=b114b3960c47b641a85eac44d974a6e71921613bee85273e3d7de811ba4e937e`
- 认证执行仅允许 `S1_DETERMINISTIC_MULTI_START`；S2 不得用于失败救场。

本轮在任何求解前冻结了 30 帧 SparseValidationV2（HIGH/MID/LOW 各 10 帧）、五帧 determinism subset 和 `SparseValidationV2GateContractV1`。全部样本与 106 帧 method-development exclusion ledger 零重叠。

## 结果

SparseV2 的技术完成为 30/30，20 个 old-above-target 帧全部恢复到 `E_IM <= 1e-4`，中位相对下降为 54.44%，灾难性回归为 0，五帧 determinism subset 的三次重复一致。

冻结 gate 仍判定 `SPARSE_VALIDATION_V2=FAIL`：全体样本 interaction non-regression 只有 20/30（66.67%），低于预注册的 90%。10 个 LOW control 均继续满足 `E_IM <= 1e-4`，但每个均相对旧轨迹产生小幅 `E_IM` 上升，因此触发 overall non-regression 硬条件。

按状态机立即硬停：

- `WINDOW_VALIDATION_V2=NOT_RUN`
- `DEV2_FRAME0_HARD_CONTROL=NOT_RUN`
- `FULL_DEV2_COMPUTE_AUTHORIZED=NO`
- `DEV2_FULL_PRODUCTION_SOLVE_COUNT=0`
- DEV1 full refinement、O6、Support、PhysX、frozen evaluation、PPO 均未运行

不得通过增加预算、添加 seed、切换 S2、改变 threshold/retention/fallback 后在同一 SparseV2 上补考。未来若进行方法修改，必须创建新版本并使用新的 untouched validation。

## CLI

```bash
conda run -n toporetarget-rl \
  python scripts/data/run_oakink2_o5rd2d.py --help
```

当前 authoritative evidence 位于：

```text
.local/reports/oakink2_o5rd2d_objective_v2_independent_certification_v1/
```
