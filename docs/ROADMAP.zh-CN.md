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
final refinement（均保留显式假设）；阶段 10 的 bounded reference-runtime milestone 已接受，preferred 性能、production 和 real-time 范围仍开放；阶段 11–19 尚未开始，不能把规划内容描述为已实现算法。

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
   几何与 SDF（有界完成，保留显式假设）；7. 相对骨方向初始化；8. 共享交互图与 Laplacian 坐标（有界完成，保留显式假设）；9. 带 slack 的受限
   优化；10. GRAB→Arti-MANO 端到端重定向；11. 指标与 ContactPose；12. OakInk、DexYCB、
   HO-Cap；13. ARCTIC、OakInk2、TACO；14. 任意灵巧手 URDF/MJCF 插件；15. baseline 与消融；
   16. reference-tracking PPO；17. 论文实验；18. 性能优化与 v1.0；19. MANO 清理、SPIDER
   等非论文扩展。

阶段 9 的详细边界见 `stages/STAGE_9_FINAL_CONSTRAINED_REFINEMENT.md`。

每阶段的 objective、major deliverables、definition of done 和 status 以英文路线图为准。

Wuji 三轨迹实现现已由通用 `workflow run-grab-suite` 提供；是否完成只由实验
根目录下运行生成的 `final_status.json` 决定。
