# OakInk2 O5R-B 并行诊断

`ThumbFeasibilityQualification` 是对冻结 production trajectory 的稀疏、
仅诊断实验。它固定 wrist、目标物体、全部非拇指 DOF、几何和 joint limits，
只优化由权威配置确定的 Wuji 拇指 DOF。报告 production 拇指 residual 与
确定性有限搜索找到的 `BEST_OBSERVED_FEASIBLE_THUMB_RESIDUAL`；后者不是
全局最优下界证明。

因此，若 current 到 best-observed 的差距很小，并且同时有 joint-limit 或
运动学证据，才增强 embodiment/morphology 解释；若存在大幅可实现改善，则
支持后续在稀疏帧上修复 thumb retarget。它不修改 Semantic Validity V1，也不
创建 Semantic V2。

`FirstFrameSeedAuthorityV2` 是版本化、episode-agnostic 的候选集合：neutral、
joint-range midpoint、lower quartile、upper quartile。它对无效数据 fail closed，
记录有限 probe，先按成功 probe、再按 final objective、最后按固定 tie-break
选择；正式 frame-0 仍使用冻结的 Stage-7 数值数学。它不改变 objective、
tolerance、joint limits、solver algorithm、stage 定义或 production `max_nfev`。

未来 DEV2 若获准，只能作为 `O5R-C` standalone recovery production solve，
不是原 DEV1→DEV2 warm timing benchmark 的重建；不得为 timing 重跑 DEV1。
