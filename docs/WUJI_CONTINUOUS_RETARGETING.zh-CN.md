# Wuji 连续重定向

`wuji_continuous_full_state_v1` 是固定 Wuji Hand2 Beta1 RH W1/W2/W3 suite 的工程扩展。历史 `scipy_slsqp_active_set_contact_rich_v3_fixed` profile 及其 artifact 保持 immutable，仍是 paper-core reference。

该扩展保留 Eq. (8)、碰撞约束、QuerySet 语义、bounds、slack、base prior 和全表面 audit，增加 previous-final correction transport、chart-consistent full-state temporal correction、continuity acceptance、deterministic retry 和有界五帧 receding-horizon fallback。不对完成的 trajectory 做 filter。

该 profile 不是 paper method：`paper_method=false`、`engineering_extension=true`、`author_exact=unresolved`。当前只在固定 `s1` 的 airplane、apple 和 alarm-clock 窗口验证，不构成跨 subject 或 RL-ready 结果。

固定 suite 命令见英文文档 [`WUJI_CONTINUOUS_RETARGETING.md`](WUJI_CONTINUOUS_RETARGETING.md)。所有 experiment evidence 隔离在 `.local/experiments/wuji_hand2_continuous_v1/`。

## W2.2 收口状态

有界收口见 [`stages/W2_2_WUJI_CONTINUITY_CLOSEOUT.zh-CN.md`](stages/W2_2_WUJI_CONTINUITY_CLOSEOUT.zh-CN.md)，只写入 `.local/experiments/wuji_hand2_continuous_v1/closeout_v1/`。当前状态为 `WUJI_CONTINUOUS_PROFILE_NOT_RECOMMENDED_WINDOW_FALLBACK_FAILED`：正式 W1/W2/W3 trajectory 通过数值和 continuity gate，W2 的 13 个绝对 q-step 全部由 warm/source basin 驱动；但真实 W3 五帧 shadow 返回 SLSQP status 4，center continuity gate 失败，且 W3 penetration rate 从 0.90 退化到 0.95。因此在这些 gate 解决前，不推荐该 profile 用于离线 reference generation。
