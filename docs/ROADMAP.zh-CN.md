# 路线图

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
