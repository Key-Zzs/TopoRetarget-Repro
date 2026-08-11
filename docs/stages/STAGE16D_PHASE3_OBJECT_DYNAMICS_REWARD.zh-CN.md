# Stage 16-D Phase 3 Object-Dynamics Reward

只有 V2 reference、twist observability、meaningful terminal residual、
contact-not-primary 和 physics-integrity gate 都通过，才可授权 Phase 3。
它是有界的 single-clip 实验：仅 `hocap_170650`。

## 冻结 reward

`TopoRetargetReferenceTrackingReward26DV2` 完整保留 V1 pose、link、finger、
wrist 和 26-D smoothness term，仅增加：

```text
e_v     = ||v_actual_world - v_ref_world_v2||_2
r_v     = exp(-(e_v / sigma_v)^2)
e_omega = ||omega_actual_world - omega_ref_world_v2||_2
r_omega = exp(-(e_omega / sigma_omega)^2)
```

冻结的保守权重为 `w_v=0.5`、`w_omega=0.5`，合计最大贡献为 1.0，且 contract
拒绝超过 2.0 的合计权重。冻结 scale artifact 记录 V2 pooled reference statistics
和 terminal-dynamics provenance。contact、terminal、penetration、guidance、gravity
和 clip-specific reward 均禁止。term 匹配 signed world twist，绝不只惩罚 speed
magnitude。

## Policy 与初始化

冻结的 764-D actor observation 已包含当前 6-D object twist；不允许加入 future
actual state 或改变 observation dimension。V2 reference twist 提供给带 V2
metadata assert 的 reward backend。训练从 V1 L0 checkpoint 加载 actor 和
observation normalization，但重置 critic 和 optimizer；`reward_v2_samples`
是独立计数器。

## 有界 protocol

先执行真实 host-GPU probe，并从有界 PPO smoke evidence 选择 capacity。随后运行
P1（至少 1,048,576 Reward-V2 samples），每约 1M checkpoint 做 development
evaluation，最多到 4,194,304 samples 再决定冻结 reward 是否有效。只有 4M
effectiveness rule 通过，才允许扩展到 16,777,216 samples。formal holdout 仅用于
checkpoint selection 之后的评价。

这些 sample limit 是 hard cap。常规 update 使用冻结的 40-step rollout；若目标还
差少于 40 个、且与每个环境对齐的 control step，则最后一次 PPO update 精确使用该
短 rollout，使记录的 `reward_v2_samples` 到达目标而不超额。

Evaluation 将 source-relative geometry 保留为 diagnostic。absolute runtime
geometry safety、contact causality、no hidden control、terminal contact 与
terminal stability 始终是独立 gate。
