# OakInk2 O5R-D2A RetargetObjectiveV2 设计结论

## 决定

```text
RETARGET_OBJECTIVE_V2_DESIGN_STATUS=NO_CANDIDATE_READY
SELECTED_OBJECTIVE=NONE
RETARGET_OBJECTIVE_V2_SHA256=null
```

O5R-D2A 修复了 objective development 使用的 production context binding，并比较了两个 interaction-first 候选。但没有候选能在全部 development windows 上同时满足冻结的 non-regression contract，因此本轮没有冻结 ObjectiveV2，也不允许进入 independent validation 或任何 downstream 阶段。

## Context binding

`ProductionObjectiveContextBindingV2` 显式绑定 active/current source frame、previous source/runtime frame、previous robot state、continuous predicted base/q、base-correction reference、object pose、robot/side/DOF mapping 和 source hand。frame 0 保留 V1 历史上的 no-temporal branch。

在全部 55 个已消费 Q0 states 上，V2 binding 加未修改的 V1 公式精确复现所有 component 和 total：

```text
PRODUCTION_OBJECTIVE_CONTEXT_BINDING_V2=PASS
N=55
max total absolute difference=0.0
max component absolute difference=0.0
```

## 候选定义

Candidate A 先最小化 per-frame `E_IM`，再在保留 primary 结果的约束下最小化去除 interaction 后的 V1 fidelity terms。它在低于 Semantic V1 target 后仍持续施加 interaction pressure，而且 technical completion 更差，因此被拒绝。

Candidate B 先最小化无量纲 hinge：

\[
\max(E_{IM}/10^{-4}-1,0)^2,
\]

再在 interaction-retention 约束下最小化去除 interaction 后的 V1 fidelity terms。达到或低于冻结 target 后，它的 interaction pressure 严格为零。Candidate B 有希望，但尚未 ready。

两个候选都保留 production collision constraints 和 Wuji joint bounds。wrist/bone limits 来自冻结的 Semantic V1；continuity limits 来自 `wuji_continuous_full_state_v1`。没有任何 threshold/budget 根据 DEV1 Q1 outcome 拟合，也没有进行 weight grid search。

## Development evidence

- B1 的 25 个 Q1 endpoints 全部改善 interaction 并且 collision-feasible，但全部违反既有 continuous-profile correction limits。Candidate B 在 7 个 LOW controls 上都正确饱和；Candidate A 没有。
- 15 个 deterministic single-frame cases 中，Candidate B 完成 15/15，没有新增 hard validity violation，并使 20% 的 above-target cases 进入 target。
- 3 个 deterministic 20-frame sequential windows 新增 51 个 development frames；future exclusion ledger 共有 106 个 unique DEV1 frames。
- Candidate B 把 MID-window p95 从 `1.379943803e-4` 降到 `9.710502070e-5`，同时保持 LOW window 低于 target。
- Candidate B 的 HIGH window 仍不完整：在 continuous non-regression budget 下只有 10/20 selected states technically usable，p95 仍为 `1.939272528e-4`。

未修改的 Semantic V1 evaluator 在 Candidate B 的 MID/LOW development windows 上通过，在 HIGH window 上因 interaction geometry 失败。这些只是 development compatibility evidence，不是 trajectory validation。

## 执行边界

Objective V1、Semantic V1、StructuredSolverV1、Manifest V2、Split V2、Wuji/MANO authorities 和旧 DEV1 trajectory 均未修改。本轮 DEV1 full rerun/refinement 为 0，DEV2 solve 为 0；没有创建 SparseValidationV2 或 WindowValidationV2，也没有消费 heldout/certification data。

下一步只能继续进行 bounded objective/search design，解决 Candidate B 的 HIGH-window continuity/technical failure。在 candidate 预先冻结、并随后选择全新零重叠 sparse/contiguous-window sets 之前，independent certification 保持 blocked。
