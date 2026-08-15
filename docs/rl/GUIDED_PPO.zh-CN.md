# 用于 Assisted Simulation Data 的 Guided PPO

这是 opt-in assisted-data workflow，不是 causal physical PPO，也不是论文作者原样的
TopoRetarget RL。必须标记为：

```text
ENGINEERING_EXTENSION_ASSISTED_DYNAMICS
external_guidance=true
assisted_dynamics=true
causal_physics=false
```

选定的 `ObjectGuidanceContractV1` 向 dynamic PhysX object 注入有界、可审计的 reference
wrench。它不进入 764D observation、26D policy action、Reward V3、Reward V4、reference 或
controller。

guided run 必须使用已复制并完成 SHA-256 校验的 Reference Kinematics V2 输入。`mode=none`
是默认值且严格为零，兼容历史 zero-guidance trace loading；`mode=reference_wrench_v1` 只能
使用 V2 的 corrected timestamps 与 pose-derived world twists。

PPO 前先运行 G2，并在两条 clip 与两个 reward mode 上选择唯一全局 G3 profile。assistance
metrics 仅用于报告，不能改写历史 V3/V4 checkpoint ranking。guided export 保留 exact contact
telemetry，并新增 wrench/error/clipping/active provenance。最终比较必须同时包含 interaction、
twist、penetration、Evaluation Suite V2 和 assistance dominance，不能只看 success rate。
