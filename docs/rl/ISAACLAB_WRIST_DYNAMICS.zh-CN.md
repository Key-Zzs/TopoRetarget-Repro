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

## Fail-closed 结果

C.3 wrist gate 为最多 2 cm、10 度。最终共享 41-step profile 为 3.35 cm、23.00 度。
增加到 100 N/6 Nm 更差（8.53 cm，83.3% force saturation）。因此 C.3 wrist dynamics 为
`FAIL`；不得运行下游 benchmark、oracle 或 PPO。

本机证据位于 `.local/reports/stage16c3_repair_c5_oracle/wrist_*.json`。
