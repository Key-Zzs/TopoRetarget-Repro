# Isaac Lab wrist dynamics 诊断

## 范围与冻结边界

这是仅用于 C.3 的工程诊断。它保留两条不可变的 41-frame、20 Hz world-wrist
reference、26-D action 与 C.2 observation contract；不授权 C.4、C.5、PPO 或真实
控制结论。

## 已确认的事实

- 控制器在每个 120 Hz substep 采样：translation 使用 cubic Hermite，orientation
  使用 shortest-arc SLERP，首尾样本精确等于冻结的 20 Hz key。
- wrench 每个 substep 通过 instantaneous composer 写入，不会在旧 wrist pose 上保持
  过期的 world-to-link 转换。
- baseline-subtracted 的真实 PhysX ±x/±y/±z force/torque probe 通过。root quaternion
  为 `wxyz`，root twist 是 world-frame，正向 world wrench 有预期的带符号响应。
- F0（无 finger drive）、F1（zero target）、F2（reference target）确认 response 是耦合的。
  F2 static body-frame matrix 只保留在 V3 作诊断，未接受为 trajectory control。

## C.3R2--C.5 fail-closed 结果

C3-0 的 fully kinematic frame/reference contract 现通过 derived canonical-URDF FK
target 验证，frozen stored link field 保持不变。Path A 在 dynamic qualification 前耗尽：
5 个 reference-target response map 超过冻结的 condition-number 上限 4000。六轴
`PhysicsJoint` D6 wrapper 可以导入，但 live GPU tensor inspection 暴露 0 个 D6 joint，
因此使用明确允许的 finite virtual 3P+3R fallback。

固定 C.3 wrist gate 为 maximum 2 cm / 10 度、RMSE 1 cm / 5 度，以及 5% force/torque
saturation。三种冻结 finite virtual profile 均在两条 41-key clip 失败：conservative 为
3.91 cm/29.45 度和 4.63 cm/21.04 度；nominal 为 3.23 cm/38.34 度和 4.54 cm/37.10 度；
high authority 为 4.10 cm/53.63 度和 6.81 cm/54.38 度。有限 disturbance 保持 physical 和
finite，移除 virtual authority 后合并 position RMSE 从 0.03623 m 恶化到 0.47282 m。

最终结果为 `C3_WRIST_ACTUATION_ARCHITECTURE_BLOCKED`。没有 active profile，故无 C.2
active-profile regression。C3-1--C3-5、contact-momentum causality、C.4、C.5 和 PPO 均
fail-closed/not run。non-contact wrist gate 使用 live PhysX evolution 及 `r_wrist` 上的有界
force/torque；rollout 不写 wrist pose/velocity 或 object state，因此该 gate 不计算 immutable
task-object termination。

本机证据位于 `.local/reports/stage16c3r2_c5/`。
