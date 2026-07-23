# Stage 9.3.3 Shadow 等价性与长指退化归因

Stage 9.3.3 是围绕已接受 Stage 9.2 reference runtime 的诊断 gate。它不定义
论文方法，不改变 Eq. (1)-(9)，也不写入 Stage 9.2 final/repeat/checkpoint 或
Stage 10 export artifact。

流程先绑定 manifest、canonical source、Stage 7 warm state、Stage 8 graph、
object pose、512 个 collision samples、QuerySet、正式 solver/execution profile、
论文权重、canonical reference SDF，以及 `t-1` 的正式 Stage 9.2 final state。
随后对 official profile 做至少 3 次独立 repeat。contract
`toporetarget.shadow_equivalence.v1` 使用
`max(float64_floor, 20 * repeat pairwise max)`，并预先固定 hard cap；不能把毫米
级差异反向放宽成 float noise。status 9、strict acceptance 失败、identity/context
不一致都会 fail-closed；只有 `EXACT` 或 `NUMERICALLY_EQUIVALENT` 才能进入 shadow。

六个 shadow profile 是 official、half-margin、zero-margin、full-512、minimal
soft-safe projection 和 official-slack projection。全部 profile 共用 canonical
reference-winding 评价；projection 不是正式 trajectory，不允许跨不同 objective
直接排名。每个 frame/profile/repeat 都有独立原子 diagnostic checkpoint，支持
`--resume` 和 `--max-wall-time`，但 checkpoint 不是正式 Stage 9 artifact。

当前 `[49,10,14]` 三个 local frame 的 final QuerySet IDs/order 和 strict feasibility
相同，但 qpos 最大差异约为 `1.526e-3`、`1.348e-4`、`9.306e-5` rad，canonical
full-512 SDF 差异约为 `6.291e-5`、`2.768e-5`、`1.137e-5` m；同时 manifest 的
`git_commit=23e6465` 与 runtime environment 的 `58fa77c` 不一致。因此当前结果是：

```text
SHADOW_BASELINE_NOT_NUMERICALLY_EQUIVALENT
RETURN_TO_STAGE9_3_2_SHADOW_HARNESS_FIX
ENTER_STAGE9_4=NO
mandatory shadow profiles run=0
```

证据位于 `.local/runs/stage9_3_3_shadow_equivalence/<run>/`。baseline gate 失败时，
ablation 命令只写入 blocked `shadow_manifest.json`，不会运行任何 shadow profile。
