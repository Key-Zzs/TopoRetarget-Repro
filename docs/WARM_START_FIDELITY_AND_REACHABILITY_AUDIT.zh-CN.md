# Stage 7.1——Warm-Start 保真度与可达性审计

Stage 7.1 是已接受 Stage 10 reference runtime 与后续 Stage 9.3.3 工作之间的只读审计边界。
它不会重新生成或覆盖 Stage 7、Stage 8、Stage 9.2、Stage 9.3.2 或 Stage 10 artifact。
正式 Stage 7 目标仍然只是相对骨方向目标：

```text
E_bone = 15 个相邻 pair 的 ||f_robot - f_source||^2 之和
E_2    = lambda_warm * E_bone + lambda_smooth * ||q_t - q_(t-1)||^2
```

接触、表面距离、object-relative 几何和 Stage 8 Laplacian fidelity 单独报告；contact-retention
proxy 不是 ground truth。base seed 仍使用显式工程约定
`T^S_B = T^S_Hs (T^B_Hr(q))^-1`，不能静默写成论文事实。

## 重现命令

使用隔离的 Python 3.12 环境和 manifest 驱动的 accepted run：

```bash
env PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/deepcybo/miniconda3/envs/topo-retarget/bin/python \
  -m toporetarget workflow audit-warm-start \
  --run .local/runs/stage10_reference_runtime/s1__airplane_lift__right__artimano_rh__f000240_f000300/manifest.json \
  --canonical-contact-audit .local/runs/stage9_3_2_canonical_reaudit/s1__airplane_lift__right__artimano_rh__f000240_f000300 \
  --output-root .local/runs/stage7_1_warmstart_audit/s1__airplane_lift__right__artimano_rh__f000240_f000300 \
  --html --run-reachability-diagnostics --diagnostic-frames auto
```

只读 pass 会用持久化 qpos 重放并重算 frame、FK anchor、bone pair、Eq. (1)/(2)、base alignment、
per-finger attribution、Stage 8 evaluation 指标以及 source/warm/final contact proxy。同时审计
source MediaPipe semantic chain、Arti-MANO anchor link、thumb URDF ancestry/axis、joint limits、
local Jacobian observability 和只读 base-alignment alternatives。

## 可达性解释

diagnostic solve 只使用 5 个有界代表帧。diagnostic root 中包括 Stage 7 formal replay、thumb-only
和 all-joint canonical-keypoint fit、thumb formal-feature fit、fixed-base/base-adjusted fit、
no/reduced temporal-weight 对比，以及确定性的 4096 点 Sobol thumb workspace sample。
diagnostic IK 不是论文方法，不是 accepted reference，也不能写入正式 artifact。

workspace report 同时记录 raw source target、robot-length reconstructed target、thumb tip/pad
samples、nearest distance、direction error、sampled convex-hull membership，并明确 sampled
workspace 不是严格的全局可达性证明。robot-length target 只用于 morphology 诊断，不能替换 source
trajectory。

## 当前 accepted run 结果

对 `s1/airplane_lift`、right Arti-MANO RH、local `[0,60)` / global `[240,300)`，当前结果是：

- `WARM_START_FORMALLY_VALID_CONTINUE_STAGE9_3_3`；
- `CONTINUE_STAGE9_3_3=YES`；
- persisted Stage 7 replay gates 全部通过，最大重算差异约 `4.4e-16`，official solver invocation 为 `0`；
- source mapping、robot anchor mapping、frame 和 base-seed gates 通过；
- raw thumb target 最近 sampled-workspace 距离平均约 `12.51 mm`，robot-length reconstructed target 平均约
  `3.81 mm`，且所有 selected frame 都在 5 mm diagnostic proximity 内；
- whole-hand final canonical-keypoint RMSE 与报告的 `E_IM` 高于 warm，因此 final refinement degradation
  作为独立 ranked cause 保留；
- 45 次 solver invocation 全部为 diagnostic-only，且只写入
  `.local/runs/stage7_1_reachability_diagnostics/`。

morphology 结果是 embodiment gap 解释，不是修改正式 Stage 7 数学的证据。final-retargeting trade-off
仍属于 Stage 9 问题；Stage 9.3.3 可以继续，Stage 9.4 仍需后续 readiness 判断，不能推导 physics/RL ready。

## 输出契约

audit root 包含 `stage7_1_summary.json/.md`、`stage7_1_readiness.json`、`stage7_artifact_replay.json`、
mapping/frame/base audits、per-finger 与 warm-vs-final attribution、joint-limit/Jacobian reports、
`root_cause_analysis.json`、`warmstart_fidelity_and_reachability.html`、`html_headless_smoke.json` 和
`official_artifact_immutability.json`。diagnostic root 单独包含 `diagnostic_manifest.json`、
`reachability_results_per_frame.csv`、profile JSON、`thumb_workspace_audit.json` 和
`thumb_workspace_points.npz`。

所有正式输入的 hash 与 mtime 必须保持不变。`official_artifacts_changed` 非零、official solver invocation
非零、mapping/replay 失败或出现非预期 Git worktree 改动，都使 readiness 结果无效。
