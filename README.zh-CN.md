# TopoRetarget-Repro

[English README](README.md)

TopoRetarget-Repro 是对
[*TopoRetarget: Interaction-Preserving Retargeting for Dexterous Manipulation*](https://arxiv.org/abs/2606.16272)
的独立、可追溯复现。它通过规范 HOI 合同、MANO 语义转换、目标手运动学、几何/SDF
处理、交互感知优化、验证和绑定 manifest 的导出，将手-物交互运动转为可审计的灵巧手参考轨迹。

## 研究目标与边界

项目研究物理可行的灵巧手仿真，并保持以下因果链：

```text
robot action -> hand-object contact -> object dynamics
```

当前主线是 PPO-26D 物理修正。rollout 中不得使用 object guidance force、隐藏 object
controller、object pose/velocity write、attachment 或 suction。未来 H2R assisted-data
路线与 main causal solution 分离，所有产出必须标记 `assisted=true` 和
`causal_physics=false`。

本仓库是工程复现；不声称作者级精确复现、完整数据集覆盖、实时性、硬件控制或持续的厂商支持。

## 方法总览

```text
有授权的 HOI 数据
  -> 规范 HOI 序列与坐标约定
  -> MANO / 目标手语义转换
  -> 交互感知运动学重定向
  -> 几何与接触验证
  -> 版本化机器人参考导出
  -> Isaac Lab 因果物理修正与评价
```

核心合同见 [HOI 数据](docs/HOI_DATA_INTERFACE.md)、
[坐标约定](docs/COORDINATE_CONVENTIONS.md) 和
[机器人手目标合同](docs/ROBOT_HAND_TARGET_CONTRACT.md)。

## 支持的数据与手

| 数据集 | Adapter | 说明 |
| --- | --- | --- |
| GRAB | 已支持 | 动态手-物交互序列 |
| DexYCB | 已支持 | 原生 PCA45 与 subject-shape 路由 |
| OakInk | 已支持 | 原生 hand vertices/joints 与 object transform |
| HO-Cap | 已支持 | PCA45、subject shape 与 object pose |
| ContactPose | 已支持 | 单帧静态转换 |
| ARCTIC、OakInk2、TACO | 计划中 | 尚未支持 |

| 目标手 | 运动学 | 重定向 | 碰撞 | 仿真/RL |
| --- | --- | --- | --- | --- |
| Arti-MANO | 已支持 | 已支持 | 已支持 | 不自动代表已资格化 |
| Wuji Hand2 Beta1 | 已支持 | 已支持 | 已支持 | 离线参考与因果物理主线 |
| Generic URDF/MJCF | 导入基础 | 需要 manifest | 需要 profile | 不自动代表已资格化 |

外部数据集和 MANO/SMPL-X 模型不会随仓库分发。请将授权输入、模型、生成数据、cache 和本地
run 保持在版本控制之外。

## 环境配置

通用 workflow 使用 Python `>=3.10,<3.14`；本地维护环境为 Python 3.12。Isaac Lab
使用独立冻结环境，见 [Isaac Lab direct environment contract](docs/rl/ISAACLAB_DIRECT_RL_ENV.md)。

```bash
conda create -n topo-retarget python=3.12 -y
conda activate topo-retarget
python -m pip install -U pip
python -m pip install -e ".[dev,cache,viz,grab,robot,geometry,retarget]"

export GRAB_ROOT=/path/to/GRAB
export MANO_MODEL_ROOT=/path/to/body_models/mano
export PYTHONNOUSERSITE=1
export PYTHONPATH=src
export TOPORETARGET_PYTHON="${CONDA_PREFIX}/bin/python"
```

直接设置本地路径，或从 [configs/paths.example.yaml](configs/paths.example.yaml) 开始。
`GRAB_ROOT` 必须包含已授权的数据集，`MANO_MODEL_ROOT` 必须包含已授权的 MANO 模型文件。

## Quick start 与复现入口

查看命令并验证随仓库提供的目标手资产：

```bash
"$TOPORETARGET_PYTHON" -m toporetarget --help
"$TOPORETARGET_PYTHON" -m toporetarget doctor paper
"$TOPORETARGET_PYTHON" -m toporetarget robots list
"$TOPORETARGET_PYTHON" -m toporetarget robots validate artimano_rh \
  --asset-root third_party/robot_hands/artimano
"$TOPORETARGET_PYTHON" -m toporetarget robots validate wuji_hand2_beta1_rh \
  --asset-root third_party/robot_hands/wuji_hand2_beta1
```

主要离线 pipeline 见 [configs/README.md](configs/README.md) 与 CLI help。它保留 source
数据，创建绑定 manifest 的派生产物，并将人工验收保留为明确边界。运行 paper-fidelity 检查：

```bash
"$TOPORETARGET_PYTHON" scripts/check_paper_fidelity.py
```

## 可视化入口

项目生成独立浏览器 HTML，用于查看 source、warm start、final mesh、interaction graph、
contact/collision diagnostics、continuity 和 provenance。请使用所选 pipeline manifest 给出的
可视化命令，并在浏览器中检查生成的 HTML。Isaac Lab 保存 trace 的 replay 仅用于诊断可视化，
不生成新的物理资格化结果。

## 评价

统一评价入口是 [Evaluation Suite V2](docs/rl/EVALUATION_SUITE_V2.md)。它报告 object
rotation/translation tracking、retargeted-hand joint/fingertip tracking，以及分离的
kinematic、physics 和 qualified success rate。轨迹指标使用移除 environment origin 后的共同
world/env frame；legacy metrics 仍会保留，但不会被静默重定义。

Stage 16-D 因果 PPO pipeline 支持 reference pose、object twist tracking，以及
reference-gated contact consistency reward。后者只在 policy optimization 中使用 reference
Wuji fingertip proximity 和当前 PhysX fingertip-to-active-object pair force；它不直接控制
object。

阶段性的 terminal-dynamics attribution 与详细结果进入 stage/RL 文档；machine-readable
artifact 保留在忽略的本地存储中。

## 文档索引

- [Roadmap](docs/ROADMAP.zh-CN.md) — 当前因果科研路线与未来 lanes。
- [Stage 16-D physics-consistent retargeting](docs/stages/STAGE16D_PHYSICS_CONSISTENT_RETARGETING.md)
  — physics scope、provenance 与 qualification 边界。
- [Terminal dynamics attribution](docs/stages/STAGE16D_PHASE1_TERMINAL_DYNAMICS.md)
  — Phase 1 方法与结论。
- [PPO-26D reference tracking](docs/rl/REFERENCE_TRACKING_PPO_26D.md) — action、
  observation、RSI、reward 与 gate 合同。
- [Physics-correction PPO](docs/rl/PHYSICS_CORRECTION_PPO.md) — 因果训练边界与决策树。
- [Reference-gated contact reward](docs/rl/REFERENCE_GATED_CONTACT_REWARD.md) — V3 contact
  signal 与因果边界。
- [Evaluation Suite V2](docs/rl/EVALUATION_SUITE_V2.md) — 共享指标与 success 合同。
- [Paper fidelity and engineering adaptations](docs/PAPER_FIDELITY.md) — 论文一致性与
  明确的工程适配。

## README 文档政策

README 仅作为稳定的项目入口文档。实验日志、checkpoint 记录、具体阶段指标、commit 状态和
runtime 报告禁止写入 README。详细结果进入 stage 文档和本地 machine-readable reports。

## License

见 [LICENSE](LICENSE)。请遵守上游数据集、模型、机器人资产和依赖的许可证。

## Citation

方法请引用原始 TopoRetarget 论文。使用本仓库的实现或派生产物时，请按其 release metadata
引用本仓库。
