# Evaluation Suite V2

`TopoRetargetEvaluationSuiteV2` 是 Stage 16-D PPO、未来 Multi-Clip PPO、
physical curriculum 和已支持 adapter 共用的增量 evaluation contract。它不删除
或重定义 legacy task/contact/geometry metric。

## 主指标

- `E_r`：object orientation 的 SO(3) geodesic trajectory mean，单位 degree。
- `E_t`：object-root-origin Euclidean trajectory mean，单位 centimetre。
- `E_j`：实际 Wuji keypoint 与共享 reference keypoint 的 mean error，单位 centimetre。
- `E_ft`：thumb/index/middle/ring/pinky landmark 的 mean error，单位 centimetre。

`SR_kinematic` 要求 `E_r < 30 deg`、`E_t < 3 cm`、`E_j < 8 cm` 且
`E_ft < 6 cm`。`SR_physics` 要求 terminal contact/stability、contact causality、
inter-finger 与 absolute hand-object penetration safety、action bounds、no hidden
force、no object rollout state write 和 no wrist-root teleport。source-relative
geometry fidelity 仍是独立 legacy diagnostic。`SR_qualified` 等于
`SR_kinematic AND SR_physics`。

## Reference-contact evaluation V1

Reward V3 增加 contact behavior diagnostic，但不重定义任何 success gate：reference
expected-contact fraction、actual fingertip--active-object contact fraction、expected-contact
recall、per-finger recall、unexpected-contact rate、persistent-contact recall、longest loss gap、
loss/recontact count、terminal contact/expected-contact，以及 force mean/p95/max 与 total impulse。
persistent reference-contact window 定义为任一 reference mask 连续至少 3 个 control step active。
actual contact 使用与 reward 相同的 pair-specific filtered PhysX force source 和训练前冻结的
numerical floor。

## Reference-kinematics V2 trace

Phase 3 trace 额外保存 `reference_kinematics_version=2`、signed world-frame
actual/reference object twist、residual norm 和两个冻结的 Reward V2 component。
这些字段只是额外 diagnostic，不替代 `E_r`、`E_t`、`E_j`、`E_ft` 或任一 physics
gate。若 terminal reference 仍在运动，应报告 terminal-semantics mismatch，而非
静默修改 absolute terminal-stability definition。

Reward V3 trace 还保存 reference mask、actual five-fingertip mask、精确
fingertip-object pair force、force magnitude/scale 和 `r_contact`。这些字段只是 diagnostic 与
replay field，本身不赋予 episode physics qualification。
