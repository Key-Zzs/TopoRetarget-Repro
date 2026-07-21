# 阶段 10——GRAB → Arti-MANO 端到端重定向

状态：已实现编排层；当前测试的接触丰富 `s1/airplane_lift` 窗口在既有
Stage 9 solver 处失败，因此尚未声称端到端通过。人工验收仍是独立门禁。

入口门禁：Stage 9 已关闭，门禁时 worktree/index 干净，并且
`.local/reports/stage9/manual_acceptance.json` 是人工填写的 pass，且检查帧
`0,29,59`。`pre_contact` 可以通过 Stage 9 门禁，但不能作为 Stage 10 的接触丰富证据。

Stage 10 选择一个明确的原生时间 GRAB 序列/窗口，用官方 semantic contact 和源几何
sanity 选择；通过内容签名 DAG 生成或安全复用 Stage 5–9 artifacts，检查帧数、时间戳、
左右手/机器人/物体 identity、profile hash 和 raw source integrity，生成 full-surface
audit、semantic sanity、review bundle，并在不调用 solver 的情况下导出
`toporetarget.robot_reference.v1`。

已验证的选择器结果：`[844,904)` contact-frame ratio 为 `1.0`，源接触中位距离约
`2.865 mm`，对象 mesh 严格 watertight；规格指定的 `[238,298)` 也通过；`[240,300)`
的源中位距离约 `3.046 mm`。`s7/cubemedium_inspect_1 [363,423)`、
`s1/airplane_fly_1 [729,789)` 和 `s1/cubemedium_inspect_1 [343,403)` 也通过有限显式
selector/geometry 门禁。`cubesmall [984,1044)` 在未修改的 Stage 8 strict graph 第 13 帧因
两个 simplex volume 低于 tolerance 被拒绝。其余真实 run 都完成 canonical、warm-start、
interaction graph 和 frozen interaction，但 final refinement 第 0 或第 1 帧返回既有
`scipy_slsqp_active_set_v1` 的 `Iteration limit reached`；一个 ratio=0.5 transition run
超过 40 分钟后以 SIGTERM 停止。未修改 Stage 9 solver 或其参数来掩盖任何失败。

后续有限显式查询又找到严格通过的 `s1/cylinderlarge_inspect_1 [327,387)` 和
`s1/apple_lift [1717,1777)`。cylinderlarge 通过 Stage 8 后仍在 Stage 9 第 0 帧失败；
sphere 在未修改的 strict graph 验证处失败，mug/phone/bunny 因 mesh 非 watertight 被拒绝。
只读单帧诊断显示冻结的 SLSQP profile 已得到正的 full-surface 和 residual margin，仍在
`maxiter=30` 返回 status 9；Stage 10 保留这个 fail-fast 结果。
对应的左手有限查询 `s7/cubemedium_inspect_1 [513,573)` 虽通过 selector，仍在未修改的
strict graph 第 1 帧失败，未进入 refinement。

因此在一个 contact-rich run 成功求解并通过人工 review 前，Stage 10 仍是
`implemented_pending_real_acceptance`。
