# 开发日志

本文件保存原中文 README 中的阶段状态和历史命令，作为开发快照。面向使用者的仓库概览、
环境配置、功能 workflow 和路线图请查看根目录 [README.zh-CN.md](../README.zh-CN.md)。
更详细的复现记录见 [REPRODUCTION_LOG.md](REPRODUCTION_LOG.md)。

## 阶段 3 快照状态

- 阶段 0 complete：仓库脚手架、配置、只读数据集发现和本地 Arti-MANO 导入器。
- 阶段 1 complete：完整 16 页论文审计、参数来源、assumption 和忠实度检查器。
- 阶段 2A complete：统一 HOI schema、明确坐标语义、可选 Zarr 缓存、确定性合成数据、误差
  指标和无头比较可视化。
- 阶段 2B 有界真实数据验收完成：使用用户提供的 MANO 模型和可选 SMPL-X backend 重建一条
  GRAB 序列，转换为 canonical Zarr，完成 raw-to-canonical 对比，并生成首/中/末帧可视化。
- 阶段 3 有界 source-hand adapter 完成：显式 MANO 语义到 MediaPipe 风格 21 点映射、版本化
  profile、dense/sparse regressor 路径、scene/wrist 派生视图、完整性报告、静态 PNG 与本地
  交互 viewer、合成测试和真实左右手 GRAB 验收。
- 阶段 4 完成并保留显式假设：通用 YAML robot-hand spec/registry、严格 URDF parser、可微
  Torch FK 与独立 NumPy FK、命名 qpos 和 limits、canonical MediaPipe-21-compatible 目标锚点、
  分离 visual/collision geometry、Jacobian 检查、合成 fixture，以及独立加载的 Arti-MANO
  RH/LH 验收。

当前没有实现 TopoRetarget 重定向算法、MANO→机器人 qpos 转换、数值优化、Delaunay/SDF、
RL/PPO 或 baselines。阶段 3 仍是 source-hand adapter，阶段 4 是目标手运动学接口，阶段 5
是有界数据 adapter；这些都不是完整重定向，也不声称 MediaPipe detector accuracy。

## 阶段 4 实现记录

目标手 contract 只定义 `P^r(q)`。`palm` 是工程 URDF base frame，外部 scene base pose 通过
齐次变换传入。论文精确的 wrist-centered robot frame 和 base rotation parameterization 仍
记录为 `A_ROBOT_HAND_FRAME_001`。

RH/LH tracked spec 均为 28 links、27 joints、22 actuated joints、5 fixed joints，并使用对照
两份导入 URDF 和 ManipTrans `artimano.py` 审计后的显式 22-name 顺序。共享的
`artimano_mediapipe21` profile 复用阶段 3 semantic layout，使用 link/joint origins；多轴共点
关节和 fixed fingertip joint origin 记录在 `A_ROBOT_KEYPOINT_ANCHORS_001` 与
`A_ARTIMANO_KEYPOINT_MAPPING_001` 中。

加载前先检查 imported asset manifest。本地证据为 upstream commit
`a3d08cfe3c3a5868a7f057533bcaf759c5af4705`、98 个导入文件、64 个有效 mesh reference，以及
manifest SHA-256 `c8e2c885e95cf690ec362c45e10d77cd16a60d3760efa692856617f148fe212e`。visual 和
collision geometry 保持分离；每侧分别有 21 个 visual 和 16 个 collision instance。fixed tip
link 在该资产中只有 visual，因此不生成 collision 替代物（`A_ARTIMANO_COLLISION_COVERAGE_001`）。

合成测试覆盖 parser graph error、所有支持的 joint/geometry 类型、解析 FK、batch/device/dtype、
base equivariance、named qpos、anchors、Jacobian finite difference、geometry 分离、registry
加载和 validation。opt-in 本地测试分别加载真实 RH/LH URDF。核心命令为
`toporetarget robots list|inspect|validate|fk|anchors|jacobian-check|visualize`。报告和 PNG
放在被忽略的 `.local/reports/stage4/`，不追踪任何资产文件。

下一阶段边界保持不变：没有开始 Stage 5 GRAB adapter、重定向、骨方向初始化、交互几何、
collision query、SDF 或 PPO。

有界 GRAB 读取器和真实验收记录见 [`GRAB_INSPECTION.md`](GRAB_INSPECTION.md)。这是单条明确
选定序列的片段检查，不是全量数据集转换。

## 阶段 5 实现记录

阶段 5 增加 filename-first lazy GRAB index、`GrabDatasetAdapter`、source/binary contact
modes、可选 MediaPipe21 派生、个性化 vtemp MANO 重建、原生 object/table mesh track、原子
Zarr cache、validation JSON/CSV、raw/canonical 对比和交互式 raw/canonical viewer。验收使用的
本地数据根目录由 discovery report/configuration 解析；index 包含 s1–s10 共 1,335 条 active
NPZ sequence，不导入 MANO 或帧数组。机器相关根目录只保存在被忽略的
`.local/reports/stage5/` 证据中。

真实验收序列为 `s7/cubemedium_inspect_1`，原生 120 Hz，右手和双手 `[0, 60)` clip。原生手/物体
vertices、source timestamps、contacts、个性化 `vtemp` 和 GRAB row-vector object transform
均被保留。validation 与 raw/canonical 对比通过：timestamp/translation/world-vertex error
为 0，最大 rotation error 约 `1.71e-6` 度。旧 Stage 2B cache 没有正式 native-keypoint 字段，
因此 legacy native-keypoint metric 标为 unavailable，没有用替代字段冒充通过。

交互 smoke test 覆盖 slider、callbacks、play/pause、reference、visibility toggles、artist
数量稳定和 timer 关闭。过大原生 mesh 只在 viewer 中使用 polygon fallback，canonical geometry
不变。Stage 6 以及后续 geometry、retargeting、collision、SDF 和 PPO 工作仍未开始。

viewer 还实现了只影响显示的 frame stride、播放速度、source/hand/geometry visibility controls，
以及可选 GIF/MP4 无头动画输出。本次审计使用 direct local Zarr store 读写，仍保持标准 Zarr 格式，
以适配受管文件系统；显示操作不改变 canonical schema 或 source arrays。

## 数据与本地资产

仓库不包含 GRAB、OakInk、OakInk2、ContactPose、TACO、HO-Cap、ARCTIC、DexYCB、MANO 或
SMPL-X。外部数据目录规范为：

```text
<storage-root>/<已登记数据集 alias>/data/**
```

机器相关路径只能写入被 Git 忽略的 `.local/config.yaml`，或通过环境变量设置。可参考
[`configs/paths.example.yaml`](../configs/paths.example.yaml) 和 [`.env.example`](../.env.example)。

## 历史命令

```bash
python -m pip install -e ".[dev]"
toporetarget --help
toporetarget data --help
toporetarget data make-synthetic --output .local/cache/hoi/synthetic_demo.zarr
toporetarget data inspect --input .local/cache/hoi/synthetic_demo.zarr --frame 0
toporetarget keypoints layouts
toporetarget keypoints profiles
toporetarget keypoints validate --input .local/cache/hoi/grab/cubemedium_inspect_1_rh_f000000_f000060_mp21.zarr --hand right --report .local/reports/stage3/mapping_validation.json
toporetarget keypoints visualize --input .local/cache/hoi/grab/cubemedium_inspect_1_rh_f000000_f000060_mp21.zarr --hand right --layout mediapipe21 --view scene --start-frame 0 --end-frame 60 --show --show-source-layout --show-mesh --show-labels
toporetarget doctor datasets --root "$REF2DEX_STORAGE_ROOT" --max-depth 4
toporetarget assets import-artimano --source-root "$MANIPTRANS_ROOT" --destination .local/assets/artimano
toporetarget doctor assets
toporetarget doctor paper
toporetarget doctor all
```

dataset doctor 只读扫描 registry 允许的数据集 alias，并限制目录深度、不跟随 symlink，忽略
未登记目录。Arti-MANO 命令把 ManipTrans 的完整 URDF/mesh 树导入 `.local/assets/artimano/`，
该目录不会进入 Git。论文检查命令是 `python scripts/check_paper_fidelity.py`。

## 历史开发检查

```bash
ruff check .
ruff format --check .
mypy src
pytest -q
python scripts/check_paper_fidelity.py
git diff --check
```

路线图见 [`ROADMAP.md`](ROADMAP.md)，论文审计见 [`PAPER_FIDELITY.md`](PAPER_FIDELITY.md)，
数据与许可证边界见 [`LICENSE_AND_DATA_POLICY.md`](LICENSE_AND_DATA_POLICY.md)。统一接口见
[`HOI_DATA_INTERFACE.md`](HOI_DATA_INTERFACE.md)，坐标语义见
[`COORDINATE_CONVENTIONS.md`](COORDINATE_CONVENTIONS.md)。
