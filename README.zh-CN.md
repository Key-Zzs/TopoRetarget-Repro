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
export TOPORETARGET_OUTPUT=/path/to/toporetarget-output
```

直接设置本地路径，或从 [configs/paths.example.yaml](configs/paths.example.yaml) 开始。
`GRAB_ROOT` 必须包含已授权的数据集，`MANO_MODEL_ROOT` 必须包含已授权的 MANO 模型文件。

## 快速开始与核心工作流

先完成[环境配置](#环境配置)。下面的顺序从最小 smoke check 开始，依次进入数据准备、几何重定向、
Stage16-D、评价和 replay。授权的原始数据保持只读；派生 cache、report 和 HTML 请写到仓库外，例如
`$TOPORETARGET_OUTPUT` 下。

### 1. 环境入口与最小 smoke check

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

### 2. 数据集与 reference 准备

一次只操作一条明确指定的 sequence。转换并检查一条 HOI sequence，生成绑定 manifest 的 canonical cache，
且不对 source sequence 做重采样：

```bash
"$TOPORETARGET_PYTHON" -m toporetarget data convert \
  --dataset grab --sequence <sequence-id> --grab-root "$GRAB_ROOT" \
  --mano-model-root "$MANO_MODEL_ROOT" \
  --output "$TOPORETARGET_OUTPUT/<sequence-id>.zarr"
"$TOPORETARGET_PYTHON" -m toporetarget data inspect \
  "$TOPORETARGET_OUTPUT/<sequence-id>.zarr"
```

### 3. 核心几何重定向

先检查精确参数，再让 `plan-grab`、`run-grab`、`status` 和 `validate` 指向同一条明确的 sequence/window
与 output root。该 workflow 支持恢复，且不会扫描或修改无关的 source data。

```bash
"$TOPORETARGET_PYTHON" -m toporetarget workflow plan-grab --help
"$TOPORETARGET_PYTHON" -m toporetarget workflow run-grab --help
"$TOPORETARGET_PYTHON" -m toporetarget workflow status --help
"$TOPORETARGET_PYTHON" -m toporetarget workflow validate --help
"$TOPORETARGET_PYTHON" -m toporetarget geometry --help
```

### 4. Stage16-D 因果 PPO 入口

Stage16-D 因果 PPO workflow 见 [Physics-correction PPO](docs/rl/PHYSICS_CORRECTION_PPO.md) 和
[Stage 16-D physics-consistent retargeting](docs/stages/STAGE16D_PHYSICS_CONSISTENT_RETARGETING.md)。
它仍是在冻结、简化的 zero-gravity / no-support Isaac/PhysX 合同下的因果 reference-tracking baseline：
没有 guidance force、support、attachment、隐藏 object controller，也没有 rollout-time object-state 或
wrist-root write。不声称 full-gravity 或 real-world physical validation。

后续物理路线定义 Contact-ready RSI V2、source-support feasibility 与 fail-closed 的重力/摩擦 curriculum。
其 C0--C2 pilot 已完成，但两个 reward mode 均未能同时在两个 clip 上通过全局 C2 absolute geometry gate。
因此该路线在 G3、C3/C4 与 P4 之前停止；仓库不声称 full-gravity 或 real-world validation。见
[Physics curriculum](docs/rl/PHYSICS_CURRICULUM.md)、[support
feasibility](docs/physics/SUPPORT_FEASIBILITY.md) 与 [Stage16 full-gravity causal
status](docs/stages/STAGE16_FULL_GRAVITY_CAUSAL.md)。

可复用的 source-first 支撑解析与有限平面 proxy 合同见 [Support resolution](docs/physics/SUPPORT_RESOLUTION.zh-CN.md)。当前 HOCap receipt 已通过 inferred support geometry 与 object-only full-gravity physics；runtime support transfer 仍受现有 hand-object geometry blocker 延后。

### 5. 评价、replay 与可视化

geometry inspection 是只读的；冻结 benchmark 按明确的
`inspect-datasets -> select -> freeze -> run -> evaluate` 状态机执行：

```bash
"$TOPORETARGET_PYTHON" -m toporetarget benchmark inspect-datasets --help
"$TOPORETARGET_PYTHON" -m toporetarget benchmark select --help
"$TOPORETARGET_PYTHON" -m toporetarget benchmark freeze --help
"$TOPORETARGET_PYTHON" -m toporetarget benchmark run --help
"$TOPORETARGET_PYTHON" -m toporetarget benchmark evaluate --help
```

检查已有 Isaac Lab trace。replay 仅用于诊断：不会重新训练 PPO、修改 trace，或生成新的物理资格化。

```bash
conda run -n toporetarget-isaaclab env OMNI_KIT_ACCEPT_EULA=YES \
  python scripts/rl/isaaclab/replay_stage16d_simulation_trace.py --help
```

项目生成独立浏览器 HTML，用于查看 source、warm-start、final mesh、interaction graph、contact 和
collision diagnostics、continuity 以及 provenance。请使用所选 pipeline manifest 给出的可视化命令，
并在浏览器中检查生成的 HTML。

### 6. 进一步复现

主要离线 pipeline 见 [configs/README.md](configs/README.md) 与 CLI help。完整的参数合同与验收边界见
[workflow resume and provenance](docs/WORKFLOW_RESUME_AND_PROVENANCE.md) 和
[Isaac Lab direct environment contract](docs/rl/ISAACLAB_DIRECT_RL_ENV.md)。运行 paper-fidelity 检查：

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

Stage 16-D 因果 PPO pipeline 支持 reference pose、object twist tracking 和版本化 contact reward。
**Aggregate V3 是当前稳定的默认 contact baseline**（`aggregate_v3`）。**Strict Per-Finger V4
是实验性 opt-in**（`strict_per_finger_v4`），它使用
`SourcePerFingerContactEvidenceV1`：只有 source-confirmed 或 persistent-confirmed 的指定 finger
的 MANO/object contact，才要求同名 Wuji distal/tip body 与 active object 接触。probable、
transition、proximity-only、no-contact 和 ambiguous source state 都不是 V4 的 mandatory contact
semantics。

V4 按 source-required finger 数量归一化独立 named-tip reward。因此其它 finger 的大力不能给缺失的
required finger 记分，source 要求更多 fingers 也不会改变总 contact reward scale。reward 只读取当前
经 filter 的 PhysX named-tip-to-active-object pair force，从不直接控制 object；共享 per-tip force
scale 在 PPO 前由精确的 V1 Formal20 pair-force telemetry 冻结。

已完成的 Stage16-D milestone 是在冻结、简化的 **zero-gravity / no-support** Isaac/PhysX 合同下的
物理因果 reference-tracking baseline：没有 external object guidance，也没有 rollout-time object-state
或 wrist-root write。这不是 physically realistic、real-world calibrated 或 full-gravity validation。
新配置使用稳定默认值：

```yaml
reward:
  contact:
    mode: aggregate_v3
```

如需明确 opt-in 到实验目标：

```yaml
reward:
  contact:
    mode: strict_per_finger_v4
```

阶段性的 terminal-dynamics attribution 与详细结果进入 stage/RL 文档；machine-readable
artifact 保留在忽略的本地存储中。

## 文档索引

- [Roadmap](docs/ROADMAP.zh-CN.md) — 当前因果科研路线与未来 lanes。
- [Stage 16-D physics-consistent retargeting](docs/stages/STAGE16D_PHYSICS_CONSISTENT_RETARGETING.md)
  — physics scope、provenance 与 qualification 边界。
- [Stage16-D causal zero-g milestone](docs/stages/STAGE16D_CAUSAL_ZERO_G_MILESTONE.zh-CN.md)
  — 冻结范围、稳定/默认 V3、实验性 V4 与下一物理阶段。
- [Stage 16 Physical Bootstrap](docs/stages/STAGE16_PHYSICAL_BOOTSTRAP.zh-CN.md)
  — P0/P1/P2 contract、safe-bank 边界与 P3 entry gate。
- [Physics curriculum](docs/rl/PHYSICS_CURRICULUM.md) — staged gravity/friction
  contract、global-mode selection 与 fail-closed promotion rule。
- [Stage16 full-gravity causal status](docs/stages/STAGE16_FULL_GRAVITY_CAUSAL.md)
  — 当前 P3 block 及任何 P4 claim 之前的边界。
- [Terminal dynamics attribution](docs/stages/STAGE16D_PHASE1_TERMINAL_DYNAMICS.md)
  — Phase 1 方法与结论。
- [PPO-26D reference tracking](docs/rl/REFERENCE_TRACKING_PPO_26D.md) — action、
  observation、RSI、reward 与 gate 合同。
- [Physics-correction PPO](docs/rl/PHYSICS_CORRECTION_PPO.md) — 因果训练边界与决策树。
- [Reference-gated contact reward](docs/rl/REFERENCE_GATED_CONTACT_REWARD.md) — V3 contact
  signal 与因果边界。
- [Strict per-finger contact reward](docs/rl/STRICT_PER_FINGER_CONTACT_REWARD.zh-CN.md) — V4
  source-confirmed contact semantics 与独立-finger contract。
- [Source contact semantics](docs/rl/SOURCE_CONTACT_SEMANTICS.md) — 原始 MANO/object evidence
  与冻结的 factor-eight runtime mapping。
- [Evaluation Suite V2](docs/rl/EVALUATION_SUITE_V2.md) — 共享指标与 success 合同。
- [Paper fidelity and engineering adaptations](docs/PAPER_FIDELITY.md) — 论文一致性与
  明确的工程适配。

## README 文档政策

README 文件是稳定的项目入口文档；实验日志和 run-specific metrics 位于 README 之外，进入
stage 文档和本地 machine-readable reports。

## 致谢

本仓库是对 [*TopoRetarget: Interaction-Preserving Retargeting for Dexterous
Manipulation*](https://arxiv.org/abs/2606.16272) 所述思想的独立复现与工程扩展。感谢原始作者及其研究贡献。

同时感谢为本项目提供核心基础的上游项目，包括 MANO/SMPL-X、PyTorch、Trimesh、python-fcl、MuJoCo、
NVIDIA Isaac Sim/Isaac Lab，以及本 workflow 使用的外部数据集的作者与维护者。仓库中跟踪的 Arti-MANO
快照来源于 [ManipTrans](https://github.com/ManipTrans/ManipTrans)；跟踪的 Wuji Hand2 Beta1 子集来源于
[wuji-description](https://github.com/wuji-technology/wuji-description)。

第三方数据集、人体模型、机器人资产和软件仍分别受其原始许可证与使用条款约束；本仓库许可证不会自动
重新授权这些材料。

## 引用

如果你使用了本仓库复现的 TopoRetarget 方法，请引用原论文：

```bibtex
@article{wu2026toporetarget,
  title   = {TopoRetarget: Interaction-Preserving Retargeting for Dexterous Manipulation},
  author  = {Wu, Jielin and Yao, Shenzhe and He, Guanqi and Liu, Xiaohan and Zeng, Zhaoqing
             and Jiang, Xiangrui and Yang, Han and Zhang, Wentao and Zhao, Hang},
  journal = {arXiv preprint arXiv:2606.16272},
  year    = {2026},
  doi     = {10.48550/arXiv.2606.16272}
}
```

如果你使用了本仓库提供的工程扩展、评价工具或仿真基础设施，也请同时引用本仓库：

```bibtex
@software{keyzzs_toporetarget_repro_2026,
  author = {{Key-Zzs}},
  title  = {TopoRetarget-Repro},
  url    = {https://github.com/Key-Zzs/TopoRetarget-Repro},
  year   = {2026}
}
```

该 software citation 使用仓库 metadata 中可确认的 owner identity，没有推断个人作者姓名，也没有声明 DOI。
使用外部数据集、人体模型、机器人资产或上游软件时，也请按照各自项目的说明进行引用。本地论文副本见
[docs/TopoRetarget.pdf](docs/TopoRetarget.pdf)，上游资产 provenance 见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## License

仓库代码和文档采用 GNU General Public License v3.0，见 [LICENSE](LICENSE)。Tracked 第三方资产继续保留
其上游许可证与 `third_party/robot_hands/` 中的 notices。外部 GRAB、MANO/SMPL-X 以及其它数据集/模型资源
不会在此重新分发，仍受各自条款约束。使用外部资源前请阅读
[docs/LICENSE_AND_DATA_POLICY.md](docs/LICENSE_AND_DATA_POLICY.md) 和 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
