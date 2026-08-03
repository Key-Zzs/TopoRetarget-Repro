# Stage 16-C failure-recovery 状态机

```
冻结 C2/C3 输入
  -> C3-0 reference/FK
  -> object-centric contact readout
  -> full-articulation computed torque
  -> bounded preview (TVLQR/MPC)
  -> wrist gate 通过？ -- 否 --> C3 BLOCKED -> C4/C5 NOT RUN -> PPO NOT AUTHORIZED
                       是
                         -> contact-momentum causality
                         -> 完整 C3 -> C4 -> C5
```

当前运行在 bounded preview 后 fail-closed：两档 computed-torque、独立 1/6-step
local-model holdout 与两条 41-frame MPC gate 均失败。MPC worker 的早期“退出”已确认是
reporter `KeyError` 误报，不是 CUDA/PhysX 退出。不得按结果挑选 profile、修改冻结
reference、在 rollout 中写 object/wrist state，或绕过 gate 启动 C4/C5/PPO。
