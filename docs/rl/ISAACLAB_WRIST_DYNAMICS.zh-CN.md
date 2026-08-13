# Isaac Lab wrist dynamics 诊断

## 范围与冻结边界

这是 C.3 工程诊断。C3R3/R4 保留两条不可变的 41-frame、20 Hz world-wrist reference；
C3R5 随后使用用户明确授权的单一全局 reference 时间变更：source NPZ 与 hash 不变，
在相同 20 Hz control cadence 下物化 factor-8、321-sample derived view。26-D action 与
764-D C.2 observation contract 不变；这不授权 C.5、PPO 或真实控制结论。

## 已确认的事实

- 控制器在每个 120 Hz substep 采样：translation 使用 cubic Hermite，orientation
  使用 shortest-arc SLERP，首尾样本精确等于冻结的 20 Hz key。
- wrench 每个 substep 通过 instantaneous composer 写入，不会在旧 wrist pose 上保持
  过期的 world-to-link 转换。
- baseline-subtracted 的真实 PhysX ±x/±y/±z force/torque probe 通过。root quaternion
  为 `wxyz`，root twist 是 world-frame，正向 world wrench 有预期的带符号响应。
- F0（无 finger drive）、F1（zero target）、F2（reference target）确认 response 是耦合的。
  F2 static body-frame matrix 只保留在 V3 作诊断，未接受为 trajectory control。

## C.3R3/R4 joint-dynamics 收口

C3-0 的 fully kinematic frame/reference contract 仍通过 derived canonical-URDF FK target
验证，frozen stored link field 保持不变。六轴 `PhysicsJoint` D6 wrapper 可以导入，但 live
GPU tensor 中没有 D6 joint，因此使用明确允许的串联 3P+3R articulation：3 个正交移动
关节、3 个旋转关节和冻结 Wuji hand，共 26 个 articulation DoF。固定 anchor 与小型中间
link 是抽象工程腕，不是真实机械臂。

C3R4 修正了 physics boundary 错误，但没有修改冻结 key 或 20 Hz 时间轴。每个 interval 的
6 次 pre-step controller call 现在采样 0/6 至 5/6；第六个 1/120 s physics step 后的 6/6
boundary 才精确等于 key k+1。Reset 用解析 joint reference 初始化显式腕部 qdot。运行时使用
完整 26x26 PhysX generalized mass matrix，保留 wrist-finger coupling block，并在每个
substep 读取 live Coriolis/centrifugal 与 gravity compensation。Gravity 为零，但不假设 bias
为零。

此前 bounded MPC 的“worker terminated”结论是误报：reporter 在 MPC 结果上读取
`latest["gain"]`，首个 interval 完成后触发 `KeyError`。修复异常持久化和 controller-specific
字段后，两条 worker 均完成全部 41 frame。使用 `CUDA_LAUNCH_BLOCKING=1` 的 6-substep
trace 记录了有限的 A/B、Hessian、unconstrained/projected/applied effort，以及
`apply_action`、scene write、sim step、scene update 的每个边界；Isaac Kit log 中没有 CUDA
或 PhysX 执行错误。

原 V1 identification 的高 fit R2 不能外推：独立 holdout 的 one-step / six-substep
normalized RMSE 为 0.06954/0.77685。M_ww 原始条件数为 686--1318，unit-scaled 后为
180--317，Hessian 最大为 10051。冻结的 projected-gradient step 在每个审计节点都违反
spectral stability bound，因此实现按 Hessian 最大特征值倒数限制 step，同时保持 cost、
horizon、iteration count 与 effort limit 不变。unit-scaled、逐 substep affine V2 model 的
fit R2 为 0.999959，但独立 absolute holdout 的 one/six-step RMSE 仍为
0.09453/0.62331，两项预先声明的诊断 gate 都失败。

两档 full-articulation computed-torque 均在两条 clip 上失败。最终 V2 MPC 也失败：
`hocap_170105` 的位置最大误差为 1.961 m、旋转 RMSE 为 119.13 度、最大单关节饱和率为
44.58%；`hocap_170650` 对应为 0.777 m、114.21 度、6.25%。所有 run 都保持 finite、完成
41/41 frame，且 rollout wrist/object state write 为零，但均未通过 maximum 2 cm / 10 度、
RMSE 1 cm / 5 度和 5% saturation gate。

最终状态为 `C3_EXPLICIT_WRIST_FINITE_EFFORT_TRACKING_EXHAUSTED` 与
`C3_WRIST_ACTUATION_ARCHITECTURE_BLOCKED`；没有 active controller。contact causality 与
完整 C.3 不恢复，C.4/C.5 继续 gate-blocked，PPO 未获授权。本机 C3R4 证据位于
`.local/reports/stage16c3r4_mpc_holdout_c4/`。

## C3R5 明确授权的 reference retiming

上面的 C3R4 结论继续作为原始时间轴的不可变证据，不被修复或重标。授权的 C3R5
结构性选择对两条 clip 使用同一个整数因子。source key `k` 精确保留在 runtime index
`8*k`；位置与 finger 值使用 cubic Hermite，quaternion 使用 normalized shortest-arc
interpolation，rate 缩放为 1/8。factor 2 与 4 均因 `hocap_170105` 仍未通过 finger semantic
gate 而被拒绝，factor 8 是首个全局共享通过值。gain、effort bound、action/observation
contract 与 qualification threshold 均未修改。

C3-1 object diagnostic 也恢复为历史 MuJoCo 语义：object 在 6 个 PhysX substep 内保持
dynamic，只在 interval 完成后放置到下一 reference boundary；它不会以 infinite-mass
kinematic actor 身份参与 contact solve。正式 C3-2/C3-4/contact rollout 的 object 与
wrist-root state write 仍为零。

共享 `high_authority_bounded` profile 的最终 C3-1 结果为：

| Clip | Wrist position max | Wrist rotation max | Finger final RMSE | Link final RMSE |
| --- | ---: | ---: | ---: | ---: |
| `hocap_170105` | 0.001183 m | 0.669 deg | 0.007632 rad | 0.000470 m |
| `hocap_170650` | 0.001228 m | 0.736 deg | 0.018714 rad | 0.000988 m |

两条 clip 的 wrist force/torque saturation 均为零。C3-0 至 C3-5（包括 task-level contact
causality）全部通过，因此当前状态为 `STAGE16C3_SEMANTIC_QUALIFICATION_VALIDATED`，
active controller 为 `finite_virtual_6d_wrist_actuator_v1`。证据位于
`.local/reports/stage16c3r5_reference_retiming_c4/`。
