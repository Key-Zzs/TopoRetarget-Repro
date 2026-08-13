# Stage 16-D Reference-Gated Contact Reward V3

## 状态与边界

V3 是当前 Stage 16-D causal-reward 路线。研究问题是：reference 期待 fingertip proximity 时，
奖励 actual fingertip-to-active-object contact 是否能减少 impulse-to-free-flight-to-recatch，且保持
kinematics 与 geometry safety。

因果链不变：

```text
policy action -> wrist/finger motion -> hand-object contact -> PhysX object dynamics
```

rollout 中没有 external object control。`causal_physics=true` 只表示没有 external object control，
不表示已完成 real-world physics calibration。

## Contract gate

1. Stage16DReferenceKinematicsV2 必须提供 321 个 factor-eight sample。
2. 两条 clip 必须共享同一套五指 distal-root landmark 与 force-column mapping。
3. primary reference mask 为 strict visual unsigned distance `< 0.03 m`。
4. 历史 V1 formal input 必须提供精确的 five fingertip--active-object force vector；aggregate
   force/presence 被拒绝。
5. pooled exact positive-contact median 冻结一个共享 `lambda_c`；少于 100 sample 则阻断 PPO。
6. response 必须 finite、monotonic、bounded by `w_c=1` 且 saturating。

V3 不含 contact termination。contact-loss termination、Contact-ready RSI V2、gravity/friction
curriculum 和 H2R 都明确不在本阶段范围内。
