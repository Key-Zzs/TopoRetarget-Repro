# 阶段 Q1–Q3：多数据集 benchmark、统一评价与冻结 baseline

## 范围

本阶段检查 MANO → Arti-MANO pipeline 是否能跨 contact-rich object/subject 迁移、Stage 7
morphology gap 是否重复、Stage 9 是否改变 interaction，以及两种 Eq. (9) temporal scope
在 dynamic trajectory 上的差异。不修改方法，不声称论文完整 ContactPose、RL、physics 或
real-time 结果。

## 命令

```bash
export PYTHONNOUSERSITE=1 PYTHONPATH=src
python -m toporetarget benchmark inspect-datasets --grab-root "$GRAB_ROOT" \
  --contactpose-root "$CONTACTPOSE_ROOT" --output .local/benchmarks/hoi_benchmark_v1/dataset_audit.json
python -m toporetarget benchmark select --config configs/benchmarks/hoi_benchmark_v1.yaml
python -m toporetarget benchmark freeze
python -m toporetarget benchmark run --profiles warm,scipy_slsqp_active_set_contact_rich_v2,scipy_slsqp_active_set_contact_rich_v3_fixed --resume
python -m toporetarget benchmark evaluate --metric-registry configs/metrics/hoi_metrics_v1.yaml --html
```

## 当前状态

合法状态包括 `Q1_Q2_Q3_BENCHMARK_COMPLETE`、`Q1_SELECTION_BLOCKED`、
`Q1_CONTACTPOSE_SELECTION_BLOCKED`、`Q2_METRICS_BLOCKED`、`Q3_BASELINE_EXECUTION_BLOCKED` 和
`Q1_Q2_Q3_COMPLETE_WITH_RECORDED_BASELINE_FAILURES`。最后一个只允许真实 solver/data/runtime
failure，不能用于 undersized selection。

当前本地状态为 `Q1_CONTACTPOSE_SELECTION_BLOCKED`：GRAB probe 保留 fixed clip 并选出 3 条
additional valid clip；ContactPose 110 条 candidate annotation 均没有可识别的 official
contact attribution，selected 为 0。因此没有创建 selection manifest，也没有运行 baseline。
