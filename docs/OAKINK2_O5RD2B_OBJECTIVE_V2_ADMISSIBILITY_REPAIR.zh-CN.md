# OakInk2 O5R-D2B ObjectiveV2 admissibility 修复

O5R-D2B 审计了 D2A Candidate B search 使用的 continuity 语义。
`continuous.py` 中的 `0.01 m`、`5 deg`、`0.05 rad` 用于归一化“相对 transported
prediction 的 correction”，同时也曾作为 optimizer trust region。真实 use site
不能证明它们是 physical、robot-control、Semantic V1 或实际轨迹的 hard limit。

冻结的 `RetargetSemanticValidityV1` 对实际逐帧 wrist motion 的 hard gate 是
`0.05 m` translation 和 `90 deg` rotation；它没有定义 finger-q step hard threshold。
因此 predictor correction 与 actual trajectory motion 必须分开。

D2A HIGH window 中，10 个 state 可用，另 10 个仅因 predicted finger-correction
budget 被拒绝；20 个 state 的实际 Semantic V1 translation/rotation continuity
全部通过。该证据支持 Candidate B2：保留 Candidate B interaction hinge，只把
prediction correction 降级为 soft preference/diagnostic。它不支持 Candidate C，
因为 C 分支要求主要失败来自真实 temporal continuity。

Candidate B2 在已经消费的 HIGH、MID、LOW development windows 上进行了真实
sequential evaluation。MID、LOW 达到 interaction target，但 HIGH 只有 18/20
technical completion，p95 仍超过 target；LOW 也有 1 个 technical noncompletion。
固定 development budget 没有增加。

因此最终状态为：

```text
OBJECTIVE_V2_PATH=CANDIDATE_B2
RETARGET_OBJECTIVE_V2_DESIGN_STATUS=NO_CANDIDATE_READY
RETARGET_OBJECTIVE_V2_SHA256=null
OBJECTIVE_V2_EXECUTION_CONTRACT_SHA256=null
SPARSE_VALIDATION_V2=NOT_RUN
WINDOW_VALIDATION_V2=NOT_RUN
DEV2_FRAME0_HARD_CONTROL=NOT_RUN
FULL_DEV2_COMPUTE_AUTHORIZED=NO
DEV2_FULL_PRODUCTION_SOLVE_COUNT=0
```

权威 machine-readable evidence 位于
`.local/reports/oakink2_o5rd2b_objective_v2_certification_v1/`。

