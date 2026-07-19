# TopoRetarget-Repro

[English README](README.md)

TopoRetarget-Repro 是一个非官方、独立、可追踪的
[*TopoRetarget: Interaction-Preserving Retargeting for Dexterous Manipulation*](https://arxiv.org/abs/2606.16272)
复现仓库。项目提供与机器人无关的 HOI 数据接口、明确的坐标约定、source-hand 转换工具、
复现审计，以及面向完整灵巧手重定向的分阶段路线。

当前实现已经覆盖 canonical HOI interface、有界 MANO→MediaPipe 风格 21 点 source-hand
adapter，以及 Stage 4 通用机器人手/Arti-MANO 目标运动学接口，但尚未声称实现论文完整的
机器人重定向优化器、RL pipeline 或论文实验结果。

## 仓库概览

主要入口是 toporetarget CLI。仓库按完整功能组织：

- 统一的、与机器人无关的 HOISequence 数据结构、scene-frame 几何和显式 SE(3) 变换；
- 对单条 GRAB NPZ 的只读检查，以及到 canonical Zarr 的转换；
- 显式 MANO 语义 layout 和版本化 MANO→MediaPipe21 mapping profile；
- 通用可微 URDF 手部 FK、命名 qpos、目标锚点和 Arti-MANO 左右手检查；
- source/object/timestamp 保留报告，以及静态和本地交互式几何 viewer；
- 论文忠实度审计、assumption 记录和本地 Arti-MANO 资产导入支持。

仓库不分发外部数据、MANO/SMPL-X 模型、机器人资产或提取缓存。它们应放在仓库外部，
通过 .local/ 配置。统一数据接口见
[docs/HOI_DATA_INTERFACE.md](docs/HOI_DATA_INTERFACE.md)，坐标语义见
[docs/COORDINATE_CONVENTIONS.md](docs/COORDINATE_CONVENTIONS.md)。

## TODO 与完整路线图

下面是完整的阶段 TODO。Complete 只表示该阶段文档中定义的范围完成，不表示全数据集或
论文结果级复现已经完成。

| 阶段 | 能力 | 状态 | 完成定义 / 后续 TODO |
| ---: | --- | --- | --- |
| 0 | 仓库架构与路径策略 | Complete | CLI 脚手架、配置、数据集发现和 Arti-MANO importer 通过。 |
| 1 | 论文忠实度审计 | Complete | PDF manifest、公式/表格/图追踪、assumption 和 checker 通过。 |
| 2 | Canonical HOI schema 与坐标 | Complete，有界 | Schema、lazy Zarr、对比 viewer 和有界 GRAB 检查通过。 |
| 3 | MANO→MediaPipe 风格 21 点 source adapter | Complete，有界 | Layout/profile、converter、报告、viewer、合成测试和有界真实 GRAB 检查通过；语义和 topology 假设仍显式保留。 |
| 4 | Arti-MANO 机器人适配器 | Complete，有假设 | 通用 URDF/FK 接口、显式 MediaPipe-21-compatible 锚点、分离几何检查、左右手验收、Jacobian 检查和 CLI 通过；论文 frame/mapping 假设仍显式保留。 |
| 5 | 完整 GRAB 数据集适配器 | TODO | 将当前单序列 reader 扩展为经过验证的数据集 adapter。 |
| 6 | 物体采样、碰撞几何与 SDF | TODO | 实现几何 backend 和测试。 |
| 7 | 相对骨方向初始化 | TODO | 实现并测试论文 Eq. 1–2。 |
| 8 | 交互图与 Laplacian 坐标 | TODO | 实现并测试 Eq. 3–7 的图和变形项。 |
| 9 | 带 slack 的受限优化 | TODO | 实现并测试 Eq. 8–9 的约束和优化。 |
| 10 | GRAB→Arti-MANO 端到端重定向 | TODO | 生成可复现的机器人 reference trajectory。 |
| 11 | Metrics 与 ContactPose 评估 | TODO | 实现 Eq. 10–12 指标和报告 fixture。 |
| 12 | OakInk、DexYCB、HO-Cap adapter | TODO | 添加独立验证的数据集 adapter。 |
| 13 | ARCTIC、OakInk2、TACO 扩展 | TODO | 添加独立验证的数据集 adapter。 |
| 14 | 任意灵巧手 plugin interface | TODO | 测试 URDF/MJCF hand plugin contract。 |
| 15 | Baseline 与 ablation | TODO | 添加公平的 OmniRetarget、Mink、DexPilot、GeoRT 运行。 |
| 16 | Reference-tracking PPO | TODO | 添加 RL 训练和评估 pipeline。 |
| 17 | 论文实验复现 | TODO | 复现 tables、figures、seeds 和结果报告。 |
| 18 | 性能优化与 v1.0 release | TODO | 建立 benchmark、打包和 release criteria。 |
| 19 | 非论文扩展 | TODO | 将 MANO 清理、SPIDER 等扩展单独标识。 |

维护中的路线图见 [docs/ROADMAP.md](docs/ROADMAP.md)，中文路线图见
[docs/ROADMAP.zh-CN.md](docs/ROADMAP.zh-CN.md)。

## 环境配置 Quickstart

### 依赖

- Python 3.10–3.13
- Git
- 使用对应 workflow 时才需要外部数据和模型
- 使用 --show viewer 需要图形 backend；无头 smoke test 可用 MPLBACKEND=Agg

安装当前数据和可视化 workflow 所需的完整环境：

```bash
python -m pip install -e ".[dev,cache,viz,grab,robot]"
```

只运行 core schema/test 时，可使用 python -m pip install -e ".[dev]"。

### 配置本地资源

不要把数据集或模型放入仓库。使用环境变量或被 Git 忽略的 .local/config.yaml：

```bash
export GRAB_ROOT=/path/to/GRAB                 # 包含 grab/ 和 tools/object_meshes/
export MANO_MODEL_ROOT=/path/to/MANO/models    # 包含 MANO_LEFT.pkl/MANO_RIGHT.pkl
export MANIPTRANS_ROOT=/path/to/ManipTrans     # 仅 Arti-MANO 导入需要
```

配置模板见 [configs/paths.example.yaml](configs/paths.example.yaml)，数据和许可证边界见
[docs/LICENSE_AND_DATA_POLICY.md](docs/LICENSE_AND_DATA_POLICY.md)。

### 检查安装

```bash
toporetarget --help
toporetarget data --help
toporetarget keypoints --help
toporetarget robots --help
toporetarget robots list
toporetarget doctor paper
```

## 功能 Workflow

以下按完整的用户功能组织，而不是按开发阶段组织。每节先给核心脚本，再给可选的
debug/可视化命令。

### 1. Synthetic canonical HOI

创建并检查确定性的 canonical sequence：

```bash
toporetarget data make-synthetic \
  --output .local/cache/hoi/synthetic_demo.zarr \
  --num-frames 8

toporetarget data inspect \
  --input .local/cache/hoi/synthetic_demo.zarr \
  --frame 0

toporetarget data compare \
  --dataset synthetic \
  --sequence demo \
  --canonical .local/cache/hoi/synthetic_demo.zarr \
  --layout side-by-side \
  --frame 0 \
  --output .local/reports/stage2a/synthetic_side_by_side.png \
  --error-json .local/reports/stage2a/synthetic_side_by_side.json
```

帧范围是连续的半开区间：--start-frame 0 --end-frame 60 表示 0–59 帧。
--show 是交互模式，--output 生成无头 PNG。

### 2. GRAB NPZ → canonical Zarr

GRAB reader 按单条明确序列工作：读取一个 NPZ 并选择一只手，不枚举或改写整个数据集。
生成完整序列时不要指定 --start-frame 和 --end-frame；生成片段时同时指定二者。

```bash
export GRAB_SEQUENCE="$GRAB_ROOT/grab/<subject>/<sequence>.npz"

toporetarget data describe \
  --dataset grab \
  --sequence-path "$GRAB_SEQUENCE" \
  --grab-root "$GRAB_ROOT" \
  --hand right \
  --mano-model-root "$MANO_MODEL_ROOT"

# 完整轨迹：不指定 --start-frame/--end-frame。
toporetarget data convert \
  --dataset grab \
  --sequence-path "$GRAB_SEQUENCE" \
  --grab-root "$GRAB_ROOT" \
  --hand right \
  --mano-model-root "$MANO_MODEL_ROOT" \
  --output .local/cache/hoi/grab/sequence_rh_full.zarr

# 有界检查：--end-frame 是排他的。
toporetarget data convert \
  --dataset grab \
  --sequence-path "$GRAB_SEQUENCE" \
  --grab-root "$GRAB_ROOT" \
  --hand right \
  --mano-model-root "$MANO_MODEL_ROOT" \
  --start-frame 0 \
  --end-frame 60 \
  --output .local/cache/hoi/grab/sequence_rh_f000000_f000060.zarr
```

canonical cache 包含所选手的 MANO/source 几何、wrist pose、object state、timestamps 和
provenance，但还没有 mediapipe21 track。raw/canonical 对比命令见
[docs/GRAB_INSPECTION.md](docs/GRAB_INSPECTION.md)。

### 3. MANO source trajectory → MediaPipe21 trajectory

Stage 3 converter 读取 canonical Zarr，并写入一个带有明确 mediapipe21 track 的新缓存。
它执行命名语义映射和显式 fingertip vertex mapping，不做 mirror、resample、smooth、recenter、
normalize，也不修改 source track。

```bash
toporetarget keypoints layouts
toporetarget keypoints profiles
toporetarget keypoints describe-profile \
  --profile mano_v1_2_smplx_to_mediapipe21

toporetarget keypoints convert \
  --input .local/cache/hoi/grab/sequence_rh_full.zarr \
  --output .local/cache/hoi/grab/sequence_rh_full_mp21.zarr \
  --hand right \
  --mano-model-root "$MANO_MODEL_ROOT"

toporetarget keypoints validate \
  --input .local/cache/hoi/grab/sequence_rh_full_mp21.zarr \
  --hand right \
  --layout mediapipe21 \
  --report .local/reports/stage3/sequence_rh_full_validation.json \
  --csv .local/reports/stage3/sequence_rh_full_validation.csv
```

当前 CLI 一次处理一只手。左手重复上述两个转换命令并使用 --hand left。映射 profile 和
assumption 见 [docs/MANO_TO_MEDIAPIPE21.md](docs/MANO_TO_MEDIAPIPE21.md)。

### 4. 序列可视化与 Debug

静态 PNG：

```bash
toporetarget keypoints visualize \
  --input .local/cache/hoi/sequence_rh_full_mp21.zarr \
  --hand right \
  --layout mediapipe21 \
  --view scene \
  --frame 0 \
  --show-source-layout \
  --show-mesh \
  --show-labels \
  --output .local/reports/stage3/scene_first.png
```

本地交互 viewer：

```bash
toporetarget keypoints visualize \
  --input .local/cache/hoi/sequence_rh_full_mp21.zarr \
  --hand right \
  --layout mediapipe21 \
  --view scene \
  --start-frame 0 \
  --end-frame <num-frames> \
  --show \
  --show-source-layout \
  --show-mesh \
  --show-labels
```

viewer 支持 frame slider、前后帧、scene/wrist 切换，以及 MANO mesh、source MANO joints、
MediaPipe21、skeleton edges、semantic labels、object mesh 和 axes 开关；标题显示 frame、
timestamp 和 mapping profile ID。显示变换只使用临时数组，不修改 canonical keypoint 坐标。
详细 viewer contract 见 [docs/MANO_TO_MEDIAPIPE21.md](docs/MANO_TO_MEDIAPIPE21.md)。

### 5. Arti-MANO 资产导入

从单独 checkout 的 ManipTrans 导入本地 Arti-MANO 资产树：

```bash
toporetarget assets import-artimano \
  --source-root "$MANIPTRANS_ROOT" \
  --destination .local/assets/artimano

toporetarget doctor assets
```

导入器会把 hash 和 provenance 写入被忽略的本地 manifest，不会复制 ManipTrans Python 代码。
见 [docs/UPSTREAM_REFERENCES.md](docs/UPSTREAM_REFERENCES.md) 和
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

### 6. 目标机器人手资产配置与运动学检查

Stage 4 使用已经导入的 Arti-MANO 资产作为目标手。核心检查命令如下：

```bash
toporetarget robots list
toporetarget robots inspect \
  --robot artimano_rh \
  --json .local/reports/stage4/artimano_rh_inspect.json
toporetarget robots validate \
  --robot artimano_rh \
  --report .local/reports/stage4/artimano_rh_validation.json \
  --csv .local/reports/stage4/artimano_rh_validation.csv
toporetarget robots fk \
  --robot artimano_rh --pose neutral --dtype float64 \
  --output .local/reports/stage4/artimano_rh_neutral_fk.json
toporetarget robots anchors \
  --robot artimano_rh \
  --csv .local/reports/stage4/artimano_rh_anchors.csv
```

对 `artimano_lh` 重复核心命令，以独立加载真实左手 URDF。registry list 不要求本地资产；
inspect 和 validate 从 `--asset-root`、`ARTIMANO_ASSET_ROOT`、`.local/config.yaml` 或安全的
本地默认路径解析资产根目录。

Debug/Inspection 补充命令放在核心流程之后：

```bash
toporetarget robots jacobian-check \
  --robot artimano_rh --pose random --seed 4 --dtype float64 \
  --report .local/reports/stage4/artimano_rh_jacobian.json
toporetarget robots visualize \
  --robot artimano_rh --pose neutral --geometry visual \
  --show-keypoints --show-skeleton --show-labels --show-base-frame \
  --output .local/reports/stage4/artimano_rh_neutral_visual.png
toporetarget robots visualize \
  --robot artimano_rh --pose neutral --geometry collision \
  --show-keypoints --show-skeleton \
  --output .local/reports/stage4/artimano_rh_neutral_collision.png
toporetarget robots visualize \
  --robot artimano_rh --pose random --seed 4 --geometry both \
  --show-keypoints --show-skeleton --show-labels --show-joint-axes \
  --output .local/reports/stage4/artimano_rh_random_overlay.png
```

接口会报告缺失的 collision geometry，不会静默生成替代几何。Stage 4 将 `palm` 定义为工程
URDF base frame，不选择论文中尚未确定的 wrist frame 参数化，也不执行 MANO→Arti-MANO
重定向。详见 [docs/ROBOT_HAND_INTERFACE.md](docs/ROBOT_HAND_INTERFACE.md) 和
[docs/ARTIMANO_ADAPTER.md](docs/ARTIMANO_ADAPTER.md)。

### 7. 论文追踪与复现审计

```bash
python scripts/check_paper_fidelity.py
toporetarget doctor paper
```

论文 PDF、公式/表格/图追踪和未解决假设见
[docs/PAPER_FIDELITY.md](docs/PAPER_FIDELITY.md)、
[docs/PAPER_FIDELITY.yaml](docs/PAPER_FIDELITY.yaml) 和
[docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md)。

### 8. 开发验证

```bash
ruff check .
ruff format --check .
mypy src
pytest -q
python scripts/check_paper_fidelity.py
git diff --check
```

Licensed-data 测试需要本地 GRAB/MANO 资源，并且默认不运行：

```bash
GRAB_SEQUENCE="$GRAB_SEQUENCE" \
MANO_MODEL_ROOT="$MANO_MODEL_ROOT" \
pytest -q tests/licensed_data
```

## 文档索引

- [Roadmap](docs/ROADMAP.md) / [中文路线图](docs/ROADMAP.zh-CN.md)
- [统一 HOI 接口](docs/HOI_DATA_INTERFACE.md)
- [坐标约定](docs/COORDINATE_CONVENTIONS.md)
- [GRAB 检查](docs/GRAB_INSPECTION.md)
- [MANO→MediaPipe21 adapter](docs/MANO_TO_MEDIAPIPE21.md)
- [通用机器人手接口](docs/ROBOT_HAND_INTERFACE.md)
- [Arti-MANO 目标手适配器](docs/ARTIMANO_ADAPTER.md)
- [Stage 4 报告](docs/stages/STAGE_4_ARTIMANO_TARGET_HAND.md)
- [论文忠实度](docs/PAPER_FIDELITY.md)
- [数据与许可证策略](docs/LICENSE_AND_DATA_POLICY.md)
- [开发日志](docs/DEVELOPMENT_LOG.md) / [中文开发日志](docs/DEVELOPMENT_LOG.zh-CN.md)
- [贡献指南](CONTRIBUTING.md) / [第三方声明](THIRD_PARTY_NOTICES.md)

## License

仓库代码和文档使用 GNU General Public License v3.0，见 [LICENSE](LICENSE)。外部 GRAB、
MANO/SMPL-X、ManipTrans、机器人资产和其他数据集继续遵循其自身许可证，仓库不重新分发。
使用外部资源前请阅读 [docs/LICENSE_AND_DATA_POLICY.md](docs/LICENSE_AND_DATA_POLICY.md) 和
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## Acknowledgments

本仓库感谢：

- [*TopoRetarget: Interaction-Preserving Retargeting for Dexterous Manipulation*](https://toporetarget2026.github.io/TopoRetarget/) 的作者；
- [ManipTrans 项目](https://maniptrans.github.io/)，本仓库只将其作为 acquisition-side Arti-MANO 资产来源；
- GRAB 数据集以及 MANO/SMPL-X 模型生态；

使用这些资源时请保留上游 attribution 并遵守各自条款。

## Citation

如使用本仓库或其实现说明，请引用 TopoRetarget 论文：

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

使用 GRAB、MANO/SMPL-X 或 ManipTrans 的数据、模型、资产时，也请引用相应上游项目。
本地论文副本见 [docs/TopoRetarget.pdf](docs/TopoRetarget.pdf)，上游获取说明见
[docs/UPSTREAM_REFERENCES.md](docs/UPSTREAM_REFERENCES.md)。
