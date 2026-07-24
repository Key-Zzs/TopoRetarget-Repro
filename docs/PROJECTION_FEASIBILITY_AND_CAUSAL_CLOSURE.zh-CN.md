# Stage 9.3.5 Projection 可行性与目标-约束因果闭环

Stage 9.3.5 是针对 Stage 9.3.4 current same-lineage baseline 的有界、只读诊断层。
它使用 `s1/airplane_lift`、right、`artimano_rh` 的 60 帧输入，不修改 Eq. (1)--(9)、
paper weights、Stage 7 warm-start、Stage 8 graph、historical artifact、Stage 10
manifest 或 manual acceptance。

`toporetarget.projection_state_metric.v1` 复用正式 regularization 的尺度，但把中心
改为当前帧 warm state。它明确是 `diagnostic_only`、`paper_method=false`、
`accepted_reference=false`，不是论文目标函数。official final 只有在独立的
512-sample `reference_triangle_winding` 审计通过后才作为 known feasible seed。

warm→final 路径使用 qpos/translation 线性插值和 SO(3) Exp/Log geodesic，至少采样
1001 个 alpha，记录全部可行区间并细化到 1e-6，不假设单调性。两种 projection 从
第一轮都使用全部 512 collision samples。constraint pressure 是透明的工程诊断分数，
不是 dual multiplier；counterfactual 可以不可行，且不会写成正式 artifact。

实际命令、输出目录和 Stage 9.4 人工决策边界与英文文档一致：
[`PROJECTION_FEASIBILITY_AND_CAUSAL_CLOSURE.md`](PROJECTION_FEASIBILITY_AND_CAUSAL_CLOSURE.md)。

生成的 HTML 是自包含面板，包含 frame/state 选择、warm/projection/final 切换、alpha
滑条、可行区间与逐手指 RMSE 曲线、目标端点/路径/变量组表、base 与 q 反事实、按 link
与 finger 过滤的 pressure、interaction gradient、projection attempts、branch 状态以及
root-cause/readiness 面板。显示尺度由完整报告载荷固定。`official_artifact_immutability.json`
记录 Stage 5--10/current-lineage 边界的 SHA-256 与 mtime 对账；任何变化都会使 readiness
fail closed。
