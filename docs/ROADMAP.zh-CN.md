# 路线图

本项目当前完成阶段 0（仓库创建与架构搭建）、阶段 1（论文忠实度审计）、阶段 2A（统一
HOI 数据接口）、阶段 2B 的有界真实 GRAB 检查、阶段 3 的有界 MANO→MediaPipe-style
21 关键点 source-hand adapter、阶段 4 的通用机器人手运动学接口与 Arti-MANO 目标手
适配，以及阶段 5 的有界 GRAB dataset adapter。阶段 6–19 尚未开始，不能把规划内容描述
为已实现算法。

长期阶段依次为：

0. 仓库创建与架构搭建；1. 论文忠实度审计；2. canonical HOI schema 与坐标约定；3. MANO
   到 MediaPipe 风格 21 点（有界完成，保留显式假设）；4. 通用 URDF/FK 与 Arti-MANO 目标手适配器（完成，保留显式假设）；5. GRAB 适配器（有界完成，保留显式假设）；6. 物体表面采样、碰撞
   几何与 SDF；7. 相对骨方向初始化；8. 共享交互图与 Laplacian 坐标；9. 带 slack 的受限
   优化；10. GRAB→Arti-MANO 端到端重定向；11. 指标与 ContactPose；12. OakInk、DexYCB、
   HO-Cap；13. ARCTIC、OakInk2、TACO；14. 任意灵巧手 URDF/MJCF 插件；15. baseline 与消融；
   16. reference-tracking PPO；17. 论文实验；18. 性能优化与 v1.0；19. MANO 清理、SPIDER
   等非论文扩展。

每阶段的 objective、major deliverables、definition of done 和 status 以英文路线图为准。
