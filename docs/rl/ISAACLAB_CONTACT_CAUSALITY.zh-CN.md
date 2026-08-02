# Isaac Lab contact causality gate

C.3 scene 为 C.1 的 21 个 collision-bearing hand body 各解析一个 filtered
`ContactSensor`。每个 sensor 只对应一个 hand body，只过滤两个冻结的 HO-Cap object；
virtual tip 没有 collision mesh，因此被有意排除。

这是 inventory 和 API-contract 证据，不是 contact causality。读取 21 个 filtered view
时，all-hand runtime collection 在写出 trace 前退出。该 multi-sensor 路径关闭了 contact
point 与 friction buffer，并且没有产生 force/impulse trace。因此：

- contact causality 为 `NOT_PROVEN`；
- 不把 object motion 当作 contact proxy；
- 即便 wrist gate 未来通过，C.3 也不能仅凭这些证据通过；
- C.4/C.5/PPO 仍被阻断。

失败记录位于 `.local/reports/stage16c3_repair_c5_oracle/contact_capture_status.json`。
