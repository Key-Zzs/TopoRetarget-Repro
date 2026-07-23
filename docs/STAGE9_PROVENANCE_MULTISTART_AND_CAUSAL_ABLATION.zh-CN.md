# Stage 9.3.4 provenance 重建、多起点与因果消融

Stage 9.3.4 是针对正式 `s1/airplane_lift`、右手、`artimano_rh`、全局帧
`[240,300)` 的诊断实验层。它不修改 Eq. (1)--(9)、论文权重、Stage 7 warm
artifact、Stage 8 graph、已接受 Stage 9.2 artifact、Stage 10 manifest 或人工接受结果。

historical lane 使用记录的 solver commit 和 detached worktree，并记录 Python/wheel 环境是否完全匹配；不匹配时输出 `HISTORICAL_EXACT_REPLAY_UNAVAILABLE`，不会用当前环境冒充历史重放。current lane 从 Stage 10 输入重新按 local frame 0 顺序运行 60 帧，并保留正式 solver、strict acceptance、checkpoint chain 和独立 512 点审计。

multistart 固定 objective、constraints、solver profile 和 strict acceptance，运行六种 seed：official warm、mapped previous、best feasible 以及三个确定性扰动；每个 selected frame 同时运行 official warm seed 的 frozen initial QuerySet 和 native QuerySet。

base-seed 使用无 scale、`det(R)=+1` 的 SE(3) Kabsch fit，保留 warm geometry 结果，并在 `initialization-only` 与 `seed-and-prior` protocol 下执行正式 final solve。mandatory profile 包含 official、half margin、zero margin、full 512、minimal soft projection 和 slack projection；projection 没有等价正式 solver contract 时只标记为 `PROJECTION_DIAGNOSTIC_NOT_SOLVED`。

所有输出写入 `.local/runs/stage9_3_4_*` 和 `.local/reports/stage9_3_4`，均为 diagnostic-only。接触 retention 字段是 proxy，不是 ground-truth contact；没有可靠 SLSQP multiplier 或完整 multi-frame branch-rollout 时，不生成因果证明。

最终保持 `ENTER_STAGE9_4=NO`、`HUMAN_DECISION_REQUIRED=YES`、`STOP_AFTER_STAGE9_3_4=TRUE`，等待人工审阅报告与 HTML。所有修改保持 unstaged，不执行 git add、commit、push、reset 或 tag。
