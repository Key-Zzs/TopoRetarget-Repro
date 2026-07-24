# 多数据集交互 benchmark

本文定义有界的 Q1 selection contract。版本化 schema 为
`toporetarget.hoi_benchmark.v1`，本地配置为 `configs/benchmarks/hoi_benchmark_v1.yaml`。

## Unit

GRAB unit 使用原生时间、连续 60 帧且不重采样。保留既有的 `s1/airplane_lift`、right hand、
全局 `[240,300)` unit。additional unit 由 lazy filename metadata、原生 contact labels、
personalized MANO/vtemp、source 检查和 strict reference-winding object mesh audit 选择。

ContactPose unit 是原生单个 grasp；只有 canonical adapter 需要时间维时才表示为 `T=1`，不
复制 pose 制造 trajectory。静态 temporal metric 是 `NOT_APPLICABLE`，不是零。`mug`、
`scissors` 和 Utah teapot 是 diagnostic exclusion set，不删除或修改原始数据。

## Freeze 与完整性

`benchmark_selection_manifest.json` 应记录 ID、source/object/contact hash、frame range、score、
rejection reason、Git commit 和 manifest hash；`benchmark_selection.lock` 将运行绑定到该 hash。
失败 unit 必须保留，禁止用结果替换 unit 或修改 frame range。数据身份修正必须创建新的
manifest version。

当前本地 audit 尚未 freeze：GRAB 有 1,334 条 non-fixed candidate，bounded NAS probe 实际
检查 16 条，16 条均通过声明的 contact gate；ContactPose 有 110 条 annotation candidate，
selected 为 0。110 条都没有可识别的 official contact attribution，其中 12 条同时属于
diagnostic exclusion set。状态为 `Q1_CONTACTPOSE_SELECTION_BLOCKED`，baseline 与结果级
evaluation 保持 blocked，直到提供包含 official attribution 的 ContactPose snapshot。

## Diversity 与 aggregation

selector 优先 thumb/index precision、多指支持、non-tip/palm coverage、不同 object/subject，
并在不违反 hard condition 时考虑 left-hand candidate。每个 dataset 的 macro metric 等权；
dynamic 与 static 分开汇报，ContactPose exact metric 不与 GRAB proxy 混合，也不构造单一
black-box score。
