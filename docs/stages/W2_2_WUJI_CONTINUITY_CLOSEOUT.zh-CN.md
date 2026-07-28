# W2.2 Wuji 连续重定向收口

W2.2 是冻结的 W1/W2/W3 Wuji Hand2 Beta1 RH suite 的有界诊断收口。不替换、重写或覆盖任何正式 baseline 或 continuous trajectory。

## 结果

收口状态为：

`WUJI_CONTINUOUS_PROFILE_NOT_RECOMMENDED_WINDOW_FALLBACK_FAILED`

该 profile 仍仅是离线 reference-generation 实验的工程扩展。`author_exact=unresolved`、`paper_method=false`，不声称 RL 或实时可用，也不声称跨 subject 泛化。

正式 W1/W2/W3 共 180 帧 artifact 通过数值、全表面碰撞、bounds 和 continuity 检查。W2 有 13 个绝对 q-step transition；13 个全部分类为 `SOURCE_OR_WARM_DRIVEN`，correction-driven、mixed/inconclusive 和 jump-and-return 均为 0。最大 correction q-step 为 `0.007226704582365295` rad。

## B0/B1/B2 归因

有界 ablation 使用 7 个固定窗口，并保持同一 QuerySet、碰撞 profile、paper weights 和 `maxiter=100` solver budget：

- B0：冻结 baseline、warm reset，不启用 transport 或 correction temporal；
- B1：只启用 previous-final transport，帧目标函数仍为 B0；
- B2：transport 加 correction temporal，isolated 模式禁用 retry/window。

210 个预期的 isolated/operational row 全部存在。但部分有界 solver row 失败（`B1=3`、`B2=16`），因此因果 ablation 结论为 `ABLATION_INCONCLUSIVE_DUE_TO_SOLVER_FAILURE`；该证据被保留，不转换为收益结论。

## 五帧 fallback

合成 deterministic fixture 通过 routing、checkpoint/resume 和 center-only commit。真实 W3 shadow 使用固定 global `[441,446)`、local `[34,39)`、anchor local `34`、center local `35`，future hints 为 local `36..38`。每帧都有独立 QuerySet、slack vector 和 hash。重复运行 deterministic，正式 artifact hash 未变化。

真实联合 SLSQP 返回 `status=4`（`Inequality constraints incompatible`），center 未通过 continuity thresholds，因此 window fallback gate 失败。质量 gate 还记录到 W3 penetration rate 从 `0.90` 退化到 `0.95`；该项也独立阻止推荐。

## 复现与证据

在仓库根目录执行完整收口：

```bash
export PYTHONNOUSERSITE=1 PYTHONPATH=src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
/home/deepcybo/miniconda3/envs/topo-retarget/bin/python scripts/wuji_continuity_closeout.py \
  --root .local/experiments/wuji_hand2_continuous_v1 \
  --baseline-root .local/experiments/wuji_hand2_grab3_v1 \
  --suite configs/experiments/wuji_hand2_continuous_v1.yaml
```

所有收口输出位于 `.local/experiments/wuji_hand2_continuous_v1/closeout_v1/`，重点包括：

- `w2_qstep_attribution/`；
- `bounded_ablation/` 及 42 个 solver checkpoint；
- `window_fallback/real_w3_shadow.json`；
- `reports/recommendation_gate.json` 与 `reports/artifact_integrity.json`；
- `html/` 与 `screenshots/`。

收口命令只做诊断，不执行 `git add`、commit、push、reset、clean 或 tag，也不修改 sibling `TopoRetarget-Repro-pene-loss` worktree。
