# Stage 9.3.1 signed-distance reconciliation 与有界 shadow ablation

Stage 9.3.1 是针对已接受的 Stage 9.2/Stage 10 reference runtime 与既有
Stage 9.3 contact audit 的只读对账边界。不修改 Eq. (1)-(9)、paper weights、
Stage 9.2 solver profile、accepted final artifact、Stage 10 manifest/export 或
manual acceptance，也不会运行 60 帧 optimizer。

## 命令

```bash
conda run -n topo-retarget env PYTHONNOUSERSITE=1 PYTHONPATH=src \
  python -m toporetarget workflow reconcile-contact-metrics \
  --run .local/runs/stage10_reference_runtime/<run>/manifest.json \
  --contact-audit-root .local/runs/stage9_3_contact_audit/<run> \
  --output-root .local/runs/stage9_3_1_metric_reconciliation/<run> --force

conda run -n topo-retarget env PYTHONNOUSERSITE=1 PYTHONPATH=src \
  python -m toporetarget workflow run-contact-shadow-ablation-legacy \
  --run .local/runs/stage10_reference_runtime/<run>/manifest.json \
  --reconciliation-root .local/runs/stage9_3_1_metric_reconciliation/<run> \
  --output-root .local/runs/stage9_3_shadow_ablation/<run> --frames auto
```

对账会记录每个可用输入的路径、SHA-256、mtime、profile、sequence、hand、
robot 与 frame range。shadow command 是 fail-closed 的：只有对账门禁通过时
才允许进入 solver；否则只写隔离的诊断 bundle，并保持
`solver_invocation_count: 0`。

## Metric contract

所有 signed distance 使用 positive-outside convention，单位为米。正式 Stage
9.2 的 `full_signed_distance` 是 reference triangle/winding backend 持久化的
512 点结果。`max_penetration` 定义为原始诊断量
`max(max(-min(phi_full)), 0)`，不做 tau 调整、不按显示值裁剪，也不替换成
slack。soft/hard constraint 仍分别是 `phi + tau >= 0` 与 `phi + b >= 0`。

Stage 9.3.1 会将该 reference 值与旧 Stage 9.3 报告使用的
`convex_hull_exact_solver_only` 值分开比较。在两者使用同一 SDF 定义前，旧
backend 的负值不能作为 Stage 9.2 reference acceptance 失败的证据。

## Required outputs

reconciliation 目录包含 `input_identity_audit.json`、
`signed_distance_definition_matrix.{json,md}`、`full512_identity_comparison.json`、
`full512_distance_reconciliation.{json,csv}`、mismatch records、
`transform_chain_comparison.csv`、`acceptance_replay.{json,csv}`、
`collision_offset_direction_audit.json`、`collision_offset_per_link.csv`、
`metric_reconciliation_summary.{json,md}`、`shadow_frame_selection.json`、
`metric_reconciliation_and_shadow.html` 与 `audit_manifest.json`。

input identity audit 会报告每个 artifact 的声明 schema。没有 schema marker 的
旧 NPZ/报告会明确记录为 `unversioned:<filename>`，不会伪造 paper schema。

shadow 目录包含 `shadow_manifest.json`、`shadow_frame_selection.json`、
`shadow_profiles.json`、per-frame/per-profile placeholders、causal-analysis、
comparison outputs 与 `stage9_4_readiness.json`。门禁失败时所有 profile 均为
`not_run`，并且 `diagnostic_only: true`、`paper_method: false`、
`accepted_reference: false`，不会进入 Stage 9.4。

## 当前 accepted window 结果

对于 `s1/airplane_lift`、right Arti-MANO、global `[240,300)` / local `[0,60)`：

- 512 点 identity、ordering、transform chain 与 Stage 9.2 reference replay
  通过；persisted/reference 最大差异小于 `2.5e-16 m`。
- 独立 acceptance replay 为 `60/60`，formal/replay mismatch 为零。
- 旧 Stage 9.3 backend 与 reference backend 不一致，最大绝对差异为
  `19.485 mm`，sign mismatch 为 `180`。
- visual/collision offset 方向审计结论为
  `COLLISION_VISUAL_OFFSET_DIRECTION_INCONCLUSIVE`；现有 visual mesh 是 open
  mesh，unsigned distance 不能证明 outward inflation。
- 唯一 closeout 状态为 `RETURN_TO_STAGE9_2_ACCEPTANCE_OR_METRIC_FIX`。没有
  shadow profile 运行，也没有 solver invocation。

这些结果仅是诊断证据，不会改变已接受的 Stage 9.2 Zarr、Stage 10 manifest、
exports 或 manual acceptance record。

## Stage 9.3.2 canonical 重审计边界

后续 canonical 重审计由 `workflow reaudit-contact-canonical` 执行。所有正式
contact、penetration、visual、collision、HTML、root-cause 与 shadow metric
统一使用版本化的 `reference_winding_v1` formal evaluation profile；旧报告中的
`convex_hull_exact_solver_only` 仅用于 regression disagreement，不能用于正式
acceptance 或 contact-rich 证据。详见
[`CANONICAL_CONTACT_DISTANCE_AND_REAUDIT.md`](CANONICAL_CONTACT_DISTANCE_AND_REAUDIT.md)。
