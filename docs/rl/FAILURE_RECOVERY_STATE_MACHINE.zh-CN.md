# Stage 16-C failure-recovery 状态机

```
冻结 C2/C3 输入
  -> A0 带符号的 live wrench probe
  -> A1 F0/F1/F2 identification
  -> A2 shared-profile qualification
  -> wrist gate 通过？ -- 否 --> C3 BLOCKED -> C4/C5 NOT RUN -> PPO NOT AUTHORIZED
                       是
                         -> 收集 all-hand force/impulse trace
                         -> causality gate 通过？ -- 否 --> C3 BLOCKED
                                                  是 --> C3 qualified
```

当前运行停在 A2：`A0=PASS`、`A1=DIAGNOSED`、`A2=FAIL`，且 all-hand collection
没有有效 trace。不得按结果挑选 profile、修改冻结 reference、在 rollout 中 object state
write，或绕过 gate 启动下游阶段。
