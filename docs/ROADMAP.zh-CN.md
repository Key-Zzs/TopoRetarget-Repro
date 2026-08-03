# 路线图

## 当前基础与后续顺序

M0 已完成。`main` 上的 F0 已完成：tracked Arti-MANO 资产与通用 target-hand contract 已实现并
验证，未修改 Stage 7–9 数学、正式 solver profile 或已有 Stage 10 artifact。后续顺序为：

| 里程碑 | 范围 | 状态 |
| --- | --- | --- |
| F0 main | tracked 机器人手资产与通用 target-hand contract，Arti-MANO 首个 registry 实例 | 完成 |
| P0 | 创建 `develop/pene-loss` 及其 worktree | 下一步，F0 未创建 |
| W0/W1 main | Wuji Hand2 Beta1 tracked 资产、通用注册、有界 GRAB→Wuji 验证 | 完成（有界） |
| S1 `develop/pene-loss` | 通用 SDF penetration loss | 计划 |
| I1 | 将 pene-loss 更新到最新 main，进行 Arti/Wuji × baseline/SDF 联合验证 | 计划 |
| W2/W3 | 至少三个 watertight clip 的 Wuji 重定向与 Stage 10 export | 计划 |
| R0/R1 | MJCF playback 与 PPO tracking | 后置 |
| CP | ContactPose 正式评价 | 后置 |

F0 明确不加入 Wuji Hand2、SDF penetration loss、`develop/pene-loss`、RL 或新的 artifact。详见
[`stages/F0_TARGET_HAND_FOUNDATION.md`](stages/F0_TARGET_HAND_FOUNDATION.md)。

W0/W1 现在已在 `main` 完成：左右手 Wuji Hand2 Beta1 已 tracked，并通过通用 contract 注册，
Stage 7/8/9 construction smoke 通过。W2 仍未开始，必须至少使用三个 watertight clip 完成
完整 Stage 7–9 及 contact/collision audit。S1 继续隔离在 `develop/pene-loss`，初始使用
Arti-MANO；I1 更新该分支到最新 `main` 后验证 Arti-MANO 与 Wuji 的 baseline/SDF；W3 为 export，
R0 为 MJCF playback/PD，R1 为 PPO tracking，CP 后置。

本项目当前完成阶段 0（仓库创建与架构搭建）、阶段 1（论文忠实度审计）、阶段 2A（统一
HOI 数据接口）、阶段 2B 的有界真实 GRAB 检查、阶段 3 的有界 MANO→MediaPipe-style
21 关键点 source-hand adapter、阶段 4 的通用机器人手运动学接口与 Arti-MANO 目标手
适配，以及阶段 5 的有界 GRAB dataset adapter（fresh semantic closeout 已通过）。阶段 6 已完成有界
geometry foundation（保留显式假设）；阶段 7 已完成有界相对骨方向 warm-start（保留显式假设）；
阶段 8 已完成有界 source-only 交互图和 Laplacian loss，阶段 9 已完成有界 Eq. (8)-(9)
final refinement（均保留显式假设）；阶段 10 的 bounded reference-runtime milestone 已接受，preferred 性能、production 和 real-time 范围仍开放；阶段 11 Core Contract Freeze 已完成，阶段 12–19 仍为 TODO，不能把后续规划描述为已实现算法。

新的 Q1–Q7 路线边界为：

Q1–Q3 多数据集交互 benchmark、统一自动评价和冻结 baseline 已实现为有界工程阶段；当前本地
ContactPose gate 在 freeze 前阻塞；Q4
morphology-aware warm-start、Q5 Arti-MANO surface contact proxies、Q6 contact-aware final
extension、Q7 跨轨迹自动 profile 选择均未开始。Q1–Q3 不改变现有方法，不把 GRAB proxy 当作
ContactPose 论文指标，也不把有界 ContactPose selection 当作完整 25-grasp 复现。

质量 A–E 阶段已在固定的四条 `s1` 轨迹上完成 bounded implementation：Q4 morphology-aware
warm-start、Q5 Arti-MANO surface contact proxy、Q6 contact-preserving diagnostic grid、Q7
自动 2×2 选择均保持 paper-core artifact 不变；contact extension 若未通过 gate，推荐 baseline
是合法结果。ContactPose 继续 deferred，当前结论不是 cross-subject generalization。

长期阶段依次为：

0. 仓库创建与架构搭建；1. 论文忠实度审计；2. canonical HOI schema 与坐标约定；3. MANO
   到 MediaPipe 风格 21 点（有界完成，保留显式假设）；4. 通用 URDF/FK 与 Arti-MANO 目标手适配器（完成，保留显式假设）；5. GRAB 适配器（有界完成，保留显式假设）；6. 物体表面采样、碰撞
   几何与 SDF（有界完成，保留显式假设）；7. 相对骨方向初始化；8. 共享交互图与 Laplacian 坐标（有界完成，保留显式假设）；9. 带 slack 的受限优化；10. GRAB→Arti-MANO
   端到端重定向；11. Core Contract Freeze（Canonical HOI v2、DatasetAdapter v1、RobotHandPlugin v1、RobotReference v2、MetricRegistry v1，已完成）；12. Dataset Adapter Expansion（TODO）；13. Complex HOI Expansion（DEFERRED）；14. Universal Robot Hand Plugin（DEFERRED）；15. Baseline Comparison（DEFERRED）；16. Reference Tracking PPO（16.0 功能完成；16.1a 已隔离确认 PD 通过，但当前 fixed-base/20D 接触时序动态不可行；16.2/16.3 未授权且未启动）；17. Paper Experiment Reproduction（TODO）；18. Performance Optimization（TODO）；19. Non-paper Extensions（TODO）。

阶段 9 的详细边界见 `stages/STAGE_9_FINAL_CONSTRAINED_REFINEMENT.md`。

Stage 12 的 adapter 输入已经存在，但 batch final queue 处于暂停状态；P2 v2 五帧证据完成后仍需
用户批准和长 clip qualification，checkpoint 与 exact-backend
gate 完成前不能把该阶段描述为完成或恢复批量运行。

每阶段的 objective、major deliverables、definition of done 和 status 以英文路线图为准。

## Stage 16-B world wrist-and-finger 工程扩展

Stage 16-A 仍是保留的 paper/minimal、base-relative 20D finger residual profile；其 full-approach
可控性证据保持不变。独立的 Stage 16-B 明确标记为 `ENGINEERING_EXTENSION`
`WORLD_WRIST_FINGER_TRACKING_PROTOCOL`。B.0 world reference、B.1 finite-wrench wrist 与 B.1b
逐轨迹 fixed-horizon 可控性已完成；共享 adaptive H1/H5/H10 的 B.1c 为 `PARTIAL / CLOSED`：
正式选择的 32x3 中 `170650` 20/20 通过，`170105` 0/20、进度 80%；有界 48x4
升级只把 `170105` 推进到 82.5%，仍以 5.002 cm axis error 失败。
B.2 MuJoCo single-clip PPO 为 `NOT_STARTED_GATE_BLOCKED`（0 样本、无 checkpoint）；B.3 two-clip PPO
仍为 TODO / not started；B.4/B.5
保持 TODO。它不等于真实机械臂控制、不用于 sim-to-real，也不能同论文 HO-Cap-32 结果直接比较。

Stage 16-C 路线为：C.0 Isaac Lab platform qualification 已在全部硬 gate 上通过，
状态为 `STAGE16C0_ISAACLAB_PLATFORM_VALIDATED_WITH_LIMITATIONS`；C.1 Wuji 与物体
资产迁移已完成真实 PhysX/CUDA、具名接触对及 1/128-env 验证，精确状态为
`STAGE16C1_ISAACLAB_ASSET_MIGRATION_VALIDATED`。C.2 入口已授权为
`STAGE16C2_DIRECT_RL_ENV_VALIDATED`，真实 GPU 的 1-env、交替 clip 与
128-env/1000-step smoke 均通过；显式 3P+3R high-authority candidate 的 bounded C.2
runtime-contract 1-env/128-env smoke、26 维 action basis 与 64-of-128 subset reset 均通过
（`STAGE16C2_EXPLICIT_3P3R_WRIST_VALIDATED`），但它不选择 active C.3 profile。C.3
的 stage-level 最终状态为 `STAGE16C3_SEMANTIC_QUALIFICATION_BLOCKED`，阻塞原因为
`C3_WRIST_ACTUATION_ARCHITECTURE_BLOCKED`：C3-0 derived-FK replay 与 object-centric
contact readout 各自通过，但唯一 Path A map 超过冻结的 condition-number gate，且三个
显式串联 3P+3R profile 均在两条 clip 上失败。该 fallback 暴露 6 个 GPU tensor joint，
但没有添加真实机械臂。因此 C.4 GPU vectorization 和 C.5
PhysX Oracle 均不得运行（分别为 `STAGE16C4_NOT_RUN_GATE_BLOCKED` 与
`STAGE16C5_NOT_RUN_GATE_BLOCKED`），C.6 single-clip GPU PPO 未获授权
（`STAGE16C6_SINGLE_CLIP_GPU_PPO_NOT_AUTHORIZED`；0 样本、0 checkpoint）。
C.7 two-clip GPU PPO、
C.8 dynamics randomization、C.9 geometry/MuJoCo/Isaac/PPO comparison 均为 TODO。
MuJoCo 不删除，继续负责 correctness、
deterministic regression、contact diagnostics、action replay 与 visualization；Isaac Lab 负责计划中的
GPU 并行训练。MuJoCo 与 PhysX 不要求 bitwise 一致，但必须重新完成语义对齐和 PhysX Oracle，
MuJoCo Oracle 不能直接授权 Isaac Lab PPO。当前已有 Isaac Lab 自定义 DirectRLEnv，但尚未完成
C.3 semantic qualification，也没有开始 PPO。

Stage 16-C.0 将独立平台栈冻结为 Python 3.11.15、Isaac Sim 5.1.0、Isaac Lab v2.3.2
（精确 commit `37ddf626871758333d6ed89cf64ad702aef127d0`）与 Torch 2.7.0 cu128。
其资格范围仅包括 host/install/import、有限步官方 GPU-PhysX、headless、viewer 与 128 环境证据。
C.2 已验证但 C.3 已阻断；C.4/C.5 被前置 gate 阻断，C.0/C.1/C.2 本身都不能授权 PPO。C.1 使用浮动基座 Wuji
Hand2 Beta1、两个冻结 HO-Cap free rigid object、确定性凸包代理，并验证 CUDA tensor、
128 个唯一环境原点与子集复位。无 DISPLAY 的交互 visual review 是软限制。

Wuji 三轨迹实现现已由通用 `workflow run-grab-suite` 提供；是否完成只由实验
根目录下运行生成的 `final_status.json` 决定。
## W2.1 Wuji 连续性修复

冻结的 Wuji Hand2 三轨迹 baseline 与工程 continuous profile 并存。状态图、连续性验收、重试和五帧窗口契约见 `docs/stages/W2_1_WUJI_CONTINUITY_REPAIR.md`。该里程碑不证明跨 subject 泛化，也不代表 RL-ready。

## W2.2 Wuji 连续性收口

W2.2 已完成为诊断收口，但没有完成 profile 推荐。收口包括 W2 q-step 归因、隔离 B0/B1/B2 transport-versus-temporal evidence、7 个固定 anomaly window、synthetic routing、真实 W3 五帧 shadow、deterministic replay、HTML review 和 artifact-integrity 检查。正式 trajectory 保持 immutable。记录状态为 `WUJI_CONTINUOUS_PROFILE_NOT_RECOMMENDED_WINDOW_FALLBACK_FAILED`：真实 window 返回 SLSQP status 4 且 center continuity 失败；W3 penetration-rate regression 也独立导致 quality gate 失败。见 `docs/stages/W2_2_WUJI_CONTINUITY_CLOSEOUT.zh-CN.md`。在这些 gate 解决前，不应将 continuous profile 用于离线 reference generation。
## W2.3 Wuji sequential finalization

W2.3 新增 `wuji_continuous_sequential_v1`，作为独立审计的 offline candidate。production
window fallback 被关闭，五帧 repair/shadow 与 sequential gate 隔离；冻结的 W1/W2/W3
证据会做 selected replay、多阈值碰撞 gate，并只在
`.local/experiments/wuji_hand2_continuous_v1/w2_3_finalization/` 下生成版本化 artifact。
该结果不证明 RL、real-time、cross-subject 或 author-exact validity。
# P3 compiled 歧义 spatial-FD

portable compiled CPU probe kernel 仍为实验项：五帧整体收益未达到合并门槛。

# P4 认证式 compiled exact sign

P4 增加可选的 float64 compiled generalized winding 和认证式 FD-probe 符号复用。
阈值附近仍回退到已验证 reference；在完成五帧和 60 帧验收前保持实验性、非默认，且不触及 Stage-12。
