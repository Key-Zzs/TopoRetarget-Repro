# W2.3 Wuji Sequential Finalization

W2.3 在 W2.2 的 immutable Wuji Hand2 Beta1 RH evidence 之上，新增独立命名的
sequential profile，用于离线 reference generation。它不重写 paper-core profile、正式
continuous trajectory、baseline、canonical source、warm start、interaction graph、历史
export，也不触碰受保护的 `TopoRetarget-Repro-pene-loss` worktree。

## Profile 契约

`wuji_continuous_sequential_v1` 严格派生自 `wuji_continuous_full_state_v1`；唯一的 solver
语义变化是 `window.fallback_enabled: true -> false`。metadata 将其标记为 recommended
candidate、engineering extension、offline-only，并明确
`RL_READY=NO`、`REALTIME_READY=NO`、`CROSS_SUBJECT_VALIDATED=NO`、`AUTHOR_EXACT=UNRESOLVED`。

正式 sequential path 在 propagated、trust-region 和 deterministic multi-start attempts
之后结束；五帧分支只做 diagnostic shadow。W2.3 harness 固定使用 W3 global `[441,446)`、
local `[34,39)`，local 34 是左锚点，坐标缩放为 `0.01 m / 0.1 rad / 0.05 rad / 0.001 m`，
使用 analytic block Jacobian，并且只在 scaled SLSQP 失败后启用 window-local `trust-constr`。

## Evidence 与 gate

运行命令：

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/deepcybo/miniconda3/bin/python scripts/wuji_w2_3_finalization.py
```

所有新 evidence 只写入 `.local/experiments/wuji_hand2_continuous_v1/w2_3_finalization/`。
其中包含 input identity/immutability snapshot、profile structural diff、formal execution
path audit、selected replay、受控的 W1 full replay 状态、多阈值 signed-distance penetration
审计、W3 `0.90 -> 0.95` 解释、known-feasible window oracle、deterministic shadow、版本化
NPZ/Zarr export、HTML 和最终 integrity 状态。

`R_pen(2 mm)` 是 hard paper-threshold gate：continuous 不得高于 baseline，最大深度不得
超过 2 mm，且 full-surface 与 unqueried audit 必须通过。1 mm rate、p95 depth 和最大深度
增量是 secondary warning。window 失败不会阻塞 sequential gate。

即使推荐通过，结论也只限 offline reference generation；不代表 author-exact reproduction、
real-time、RL-ready 或 cross-subject validity。
