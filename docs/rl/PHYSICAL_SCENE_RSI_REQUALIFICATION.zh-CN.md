# P3-B.6 物理场景与 RSI 再资格化

P3-B.6 是 fail-closed qualification 阶段。它对两个 HOCap clip 的全部 321
帧执行检查：根据 V2 wrist pose 与 finger state 重建完整 21 个 Wuji collision
body，使用精确的 `python-fcl` runtime manifest，并把 finite inferred table
作为 static kinematic actor 纳入检查。tracked points 与 reference ghost
只能作为诊断，不能替代正式碰撞几何。

`PhysicalReferenceValidityMaskV1` 逐帧保存 reference/source semantic、support
state、H-O/H-T/O-T/inter-finger 几何、validity、failure reasons、pose、finger
state、object twist 与 source interval。`PhysicalSafeRSIBankV1` 基于完整 bank、
冻结 geometry gate 和 support causality 构建，不包含历史 blacklist。offline
bank 只是 reset 选择证据，不授权启动 PPO。

## Runtime qualification

动态 reset 使用 Isaac Lab 5.1 / GPU PhysX、1g gravity、nominal friction、
active finite table、zero residual action，并禁止 guidance、attachment、桌面
移动以及 rollout object/wrist-root state write。每个候选使用 4 个 replica 和
20 个 control step。joint zero replay 从最早的 physically valid PRE_CONTACT
开始，只有完整跑完 320 步且没有 runtime termination 才能授权。

当前 receipt：

| Clip | Offline physical bank | Dynamic 4×20 safe states | Joint replay |
| --- | ---: | ---: | --- |
| `hocap_170105` | 162 | 66 | 未授权；step 4 `FAILURE_JOINT_LIMIT` |
| `hocap_170650` | 102 | 2 | 未授权；step 5 `FAILURE_JOINT_LIMIT` |

动态通过项均满足零 object/wrist-root rollout write；170105 的通过项还具有
持续 table contact。被拒绝的候选包含 runtime joint-limit termination，
170650 的 pre-contact 候选还未通过 table-contact/object-stability 检查。两个完整
reference trajectory 都未通过 active H-O p95 gate；`hocap_170105` 还未通过
formal H-T gate。因此唯一决策为：

```text
P3_RESTART_BLOCKED_REFERENCE_GEOMETRY
```

未启动 PPO gravity training。下一步只能是 `REFERENCE_GEOMETRY_REPAIR`；不能
通过扩大 blacklist 或移动桌面来规避问题。

完整 ignored receipt 位于
`.local/reports/stage16_p3b6_scene_rsi_requalification/`，包括 mask、safe
bank、dynamic/joint 报告、截图、`p3_restart_decision.json` 与 `handoff.md`。
