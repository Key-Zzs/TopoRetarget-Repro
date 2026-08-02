# Stage 16-C.1 Isaac Lab 资产迁移

Stage 16-C.1 是工程资产资格验证，位于已验证的 C.0 平台之后，也位于任何自定义
`DirectRLEnv`、reward、termination、PhysX Oracle 或 PPO 之前。生成的 USD 与报告
保存在被忽略的 `.local/`；Git 仅跟踪契约、配方、测试和文档。

## EULA 范围

用户明确授权本任务的 Isaac Sim 进程设置 `OMNI_KIT_ACCEPT_EULA=YES`。导入、
smoke 与渲染 CLI 仍要求显式 `--accept-eula`，并且只在对应进程内设置该变量。
此授权不包含隐私或 telemetry 数据收集同意。

## 已验证资产

- Wuji Hand2 Beta1：浮动 `r_wrist`、26 bodies、20 个 revolute joints、20/20
  逐关节响应、16/16 tracked links，以及确定性 convex collision proxy。
- HO-Cap：冻结 `hocap_170105` 与 `hocap_170650` 的 OBJ 哈希、单位尺度、质量、
  COM、惯量、摩擦、零重力、无地面与无支撑；碰撞策略为 `convex_hull_v1`。
- Runtime：真实 RTX 5080/CUDA PhysX 的 1/128-env、seeded jointwise random action、
  subset reset、具名接触对、接触位置/法向/分离量/摩擦力和有限状态证据。

质量与惯量属于工程 nominal 值，物理来源尚未解决，因此不能用于 calibrated dynamics
或 sim-to-real 声明。视觉审查单独记录，不能替代资产、接触、CUDA 或 vectorization
硬 gate。

## 状态与下一阶段边界

C.1 的精确状态为 `STAGE16C1_ISAACLAB_ASSET_MIGRATION_VALIDATED`。它只授权
C.2 入口状态 `STAGE16C2_DIRECT_RL_ENV_AUTHORIZED`；本阶段没有实现 `DirectRLEnv`，
也没有运行 PhysX Oracle 或 PPO，没有生成 PPO 样本或 checkpoint。后续实现必须作为
单独任务执行。
