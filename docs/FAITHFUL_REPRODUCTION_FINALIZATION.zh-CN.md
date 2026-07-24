# Faithful Reproduction Finalization

本阶段在 Stage 9 实现审计之后，对 `GRAB/s1/airplane_lift`、右手、
`artimano_rh`、全局帧 `[240,300)` 进行正式收口，不再新增 Stage 9.3.x
诊断阶段。

## 决策

现有数值/模型辅助证据和已完成的人类检查正式接受：**情况 A**。

- 60 帧 contact sheet 中未发现 fixed 相对旧 current-lineage 结果的可见退化；
- old→fixed 机器人关键点最大位移为 `0.654 mm`；
- base 最大平移差为 `0.276 mm`，最大旋转差为 `0.00621 rad`；
- fixed 的 60 帧全部 optimizer status=0、strict accepted，并通过 full-512
  hard/soft audit；
- old 与 fixed 的最大 raw penetration 都为零；
- fixed-old 的逐指 RMSE 变化分别为：thumb `+0.0267 mm`、index
  `+0.0336 mm`、middle `-0.0206 mm`、ring `+0.0785 mm`、pinky
  `+0.0423 mm`。

旧 Stage 9 质量门槛输出 `REPAIR_CANDIDATE_REJECTED`，原因是它要求长指至少改善
`1.5392 mm`。这是“必须显著改善”的门槛，不是肉眼退化证据。因此本次正式结论是：
论文语义修正、质量中性，不宣称质量提升。

fixed 也不是在所有连续性指标上都更平滑：最大 base translation step 和 `q`
step 略低，但最大 rotation step 与 base jerk 略高。它们仍属于亚毫米/小角度
运动，视觉检查未发现跳变，因此只能表述为连续性持平，不能宣称连续性改善。

old 和 fixed 的绝对接触保持都仍有限。source-label-conditioned 机器人视觉表面
proxy 显示，old/fixed 的 middle distal 表面与对应 source contact 区域约相距
`13 mm`。该 proxy 不是 contact ground truth，但足以阻止“修复改善了接触”的错误
表述。fixed 仅按相对旧结果质量中性验收。

## Profile 分类

权威配置为
[`configs/retarget/finalization/faithful_reproduction_profiles.yaml`](../configs/retarget/finalization/faithful_reproduction_profiles.yaml)：

- `scipy_slsqp_active_set_contact_rich_v2`：
  `historical_accepted`、非论文忠实，保留为历史工程对照；已知偏差是 base
  correction 被包含在 temporal regularization 中。
- `scipy_slsqp_active_set_contact_rich_v3_fixed`：
  canonical paper-faithful profile，状态为 `validated_quality_neutral`；
  不宣称数值质量提升。

## 论文忠实度结论

Projection 不是论文方法，只是 diagnostic-only 工具，已经封存，也不是 accepted
reference。

Eq. (9) 的实现错误是：六维 floating-base correction 被包含进 temporal `q`
项，而 base translation 和 rotation 已经各自受到独立 prior 约束。v3 fixed
只对 finger `q` 应用 temporal regularization，同时保留全部论文权重和独立
floating-base priors。

v3 fixed 现已成为 canonical faithful baseline。v2 保留为
historical/non-faithful 工程对照，因为它能够复现此前接受的工程行为，并可用于
regression。

## 新的 versioned Stage 10 候选

新候选隔离在：

```text
.local/runs/stage10_faithful_regularization_fix_v1/
  s1__airplane_lift__right__artimano_rh__f000240_f000300__faithful_regularization_fix_v1/
```

其中包括新 manifest、由 fixed artifact 导出的 NPZ/Zarr robot reference、四状态
HTML、视觉与数值审计、old-vs-new comparison、profile 分类、paper-fidelity
声明和 manual acceptance 模板。旧 Stage 10 manifest/reference 均未修改。
根目录 `INDEX.json` 将带 faithful 后缀的 run 标为唯一权威候选，并把更早的
pre-human manifest 标为 non-authoritative。

## 人工验收与最终状态

人类 reviewer 已播放 60 帧四状态 bundle，检查 local frame
`0, 9, 10, 12, 25, 27, 29, 30, 36, 39, 59`，并以所有要求的视觉检查均通过记录
情况 A。经 validator 验证的记录已写入 `review/manual_acceptance.json`。正式收口
命令为：

```bash
toporetarget workflow finalize-faithful-reproduction \
  --manual-acceptance /tmp/manual_acceptance.json
```

权威 versioned manifest 和根目录 `INDEX.json` 现均记录
`FAITHFUL_REPRODUCTION_FINALIZED_CASE_A`、`human_manual_acceptance=pass` 以及
`canonical_faithful_profile=scipy_slsqp_active_set_contact_rich_v3_fixed`。

validator 保留三种决策分支：

- A：要求所有视觉检查为 true；随后将 v3 fixed 提升为 canonical faithful
  quality-neutral profile。
- B：要求至少一个明确退化项，并填写原因和证据帧；v3 fixed 仍是论文忠实实现，
  但明确不推荐生产使用，v2 继续作为非 faithful 的历史工程对照。
- C：要求所有视觉检查为 true，并填写改善原因和证据帧；在当前 versioned export
  中记录肉眼改善，且不覆盖历史 Stage 10。
