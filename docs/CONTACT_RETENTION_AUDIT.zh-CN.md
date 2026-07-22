# Stage 9.3 接触保持与碰撞几何审计

Stage 9.3 是针对已接受 Stage 9.2/Stage 10 manifest 的只读审计流程。它从
manifest 解析 canonical、warm-start、final、interaction graph、object samples
和 collision samples；默认路径不调用 Stage 9 optimizer，也不写入正式输入
artifact。

完整 60 帧运行：

```bash
conda run -n topo-retarget env PYTHONNOUSERSITE=1 \
  python -m toporetarget workflow audit-contact-retention \
  --run .local/runs/stage10_reference_runtime/<run>/manifest.json \
  --output-dir .local/runs/stage9_3_contact_audit/<run> \
  --surface-samples 8192 --thresholds-mm 1,2,3,5,8,10 --html --force
```

审计强制使用完整 60 帧窗口，并在 `audit_manifest.json` 中记录输入解析、前后
hash/mtime、branch/HEAD、solver/execution profile、signed-distance 约定、采样
profile 以及所有输出 hash。

## 比较对象与几何约定

- **Source：** canonical 变形源手 mesh 与 21 个 MediaPipe anchor。
- **Warm-start：** 正式 warm-start artifact 中的 Arti-MANO Stage 7 初始化。
- **Final：** 已接受的 Stage 9.2 final trajectory。
- **Object：** 每帧从 object-local 变换到 scene frame 的 canonical watertight mesh。

Source 接触是诊断 proxy，不是真实接触标签。审计使用确定性的 dense mesh
sample、最近 MediaPipe21 anchor-region 和 threshold sweep。机器人保持率同时在
semantic anchor、手部 region 和 source/final threshold pair 上统计；semantic
anchor 不是 pad surface，因此另行统计 visual mesh 与独立 512 点 collision geometry。

Signed distance 采用 positive-outside、negative-inside。dense surface 数值是连续
surface 的近似并始终标注为 approximation。若正式 artifact 请求 convex-hull
solver backend，审计先用 32 个确定性 probe 与 reference triangle/SDF backend
交叉验证，记录两个 backend 和结果，不静默替换或修复 mesh。

`queryset_audit_per_point.csv` 保留 query ID、source/robot link、object-local
坐标、active margin、inclusion reason、扩展状态、slack 与 warm/final 距离。
SciPy SLSQP multiplier 被明确记录为 unavailable；改用 slack、active-set provenance
和独立 full-surface 检查。

## 目标函数、插值和可视化

`objective_tradeoff_per_frame.csv` 用相同 Stage 9 objective 定义比较 warm-start
和 final，包括 raw/weighted interaction、bone、base、temporal、slack 与 total，
不增加 contact-preservation term。

`warm_final_interpolation_per_frame.csv` 是反事实诊断路径：qpos 线性插值，base
rotation 使用 SO(3) Slerp。它不是 optimizer trajectory，不代表中间状态可行，也不
改变 final artifact。`contact_retention_proxy.json` 记录 anchor distance drift、
object-local direction consistency 和 threshold sensitivity；
`per_link_collision_visual_offset.csv` 记录 visual/collision sample 的双向近似偏移。

`trajectory_contact_audit.html` 支持 source、warm-start、final、object、visual、
collision、QuerySet、anchor、segment、threshold、frame 和 link/region 控制。HTML
只是检查工具，CSV/JSON 数值报告是权威证据。

## Shadow 与假设

默认审计不调用 solver。`--run-shadow-ablation` 是独立的诊断边界，绝不作为
paper-faithful 证据；当前实现运行的是 selected frames 上的 score-only counterfactual
分解（移除已持久化的 slack、temporal 或 base score term），不会重新生成 q，也不能
识别 optimizer 的因果效果。若未运行，`shadow_ablation_status.json` 会明确记录缺少该
反事实证据。shadow 不能替换或覆盖已接受的 Stage 9.2/Stage 10 artifact。
显式 shadow 输出隔离在 `.local/runs/stage9_3_shadow_ablation/` 下。

## Stage 9.3.1 signed-distance reconciliation 边界

完成审计后，按 [`CONTACT_METRIC_RECONCILIATION_AND_SHADOW_ABLATION.md`](CONTACT_METRIC_RECONCILIATION_AND_SHADOW_ABLATION.md)
执行只读 reconciliation。它对比 Stage 9.2 持久化的 512 点 reference SDF 与旧 Stage 9.3
报告，记录 artifact 路径、hash、mtime、点身份/顺序、transform round-trip、signed-distance
definition matrix，以及独立的 Eq. (8)-(9) acceptance replay。`reference_triangle_winding`
与旧的 `convex_hull_exact_solver_only` 必须保持分离；当定义不同，旧 backend 的负值不能直接
被解释为正式 acceptance 失败。

bounded shadow command 必须通过明确的 reconciliation gate，并隔离写入
`.local/runs/stage9_3_shadow_ablation/`。最多选择三个确定性的 representative frame。gate
失败时所有 profile 均记录为 `not_run`，solver invocation 为 0，且不会写入 formal artifact。
不能依据 unsigned offset 推断 `COLLISION_GEOMETRY_INFLATED`；只有在 mesh normal 可靠时才允许
给出方向性结论。

root-cause 报告区分 geometry inflation、collision sample coverage、semantic-anchor/
pad mismatch、QuerySet activation 以及 objective/regularization 解释，并给出置信度、
支持/反证和下一步诊断。正式 Stage 9.2 profile、weights、QuerySet、512 samples 与
strict acceptance policy 保持不变。Stage 9.3 的工程假设登记在 `docs/ASSUMPTIONS.md`。
## Stage 9.3.2 canonical re-audit

Stage 9.3.2 是独立的 v2、audit-only 边界。所有正式 contact、penetration、visual、
collision、full-512、HTML、root-cause 和 shadow evaluation 都使用版本化的
`reference_winding_v1` / `reference_triangle_winding` SDF；solver 内部允许继续使用已批准的
solver profile backend。旧 `convex_hull_exact_solver_only` 仅保留作 regression/history
diagnostic，不参与 formal pass/fail、contact-rich classification 或 readiness，依赖它的旧
Stage 9.3 负距离结论视为 superseded。

Source 和 retention 始终是 contact proxy，不是真实接触标签。Visual mesh 是开放或法向未
验证时，只能报告 unsigned coverage gap，offset direction 必须是 `INCONCLUSIVE`，不能仅凭
unsigned offset 宣称 inflated/inset。Canonical 60x512 gate 未通过时禁止 shadow solver；
shadow profile 不是论文方法，Stage 9.4 也不在本阶段实现。正式 Stage 9.2/Stage 10 artifact、
manual acceptance 和 robot export 保持不变。

完整 contract 与命令见
[`CANONICAL_CONTACT_DISTANCE_AND_REAUDIT.md`](CANONICAL_CONTACT_DISTANCE_AND_REAUDIT.md)。
