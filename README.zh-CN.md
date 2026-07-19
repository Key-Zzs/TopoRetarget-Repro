# TopoRetarget-Repro

[English README](README.md)

TopoRetarget-Repro 是 TopoRetarget 论文的非官方、独立、可追踪复现仓库基础。初始目标是
GRAB → Arti-MANO，后续目标是支持多种 HOI 数据集以及任意 URDF/MJCF 灵巧手。

## 当前状态

- 阶段 0 complete：仓库脚手架、配置、只读数据集发现和本地 Arti-MANO 导入器。
- 阶段 1 complete：完整 16 页论文审计、参数来源、assumption 和忠实度检查器。
- 阶段 2A complete：统一 HOI schema、明确坐标语义、可选 Zarr 缓存、确定性合成数据、误差
  指标和无头比较可视化已实现。
- 阶段 2B 的有界真实数据验收已完成：使用用户提供的 MANO 模型和可选 SMPL-X backend 重建
  一条 GRAB 序列，转换为 canonical Zarr，完成 raw-to-canonical 对比，并生成首/中/末帧可视化。
- 阶段 3 的有界 source-hand adapter 已完成：显式 MANO 语义到 MediaPipe 风格 21 点映射、版本化
  profile、dense/sparse regressor 路径、scene/wrist 派生视图、静态 PNG 与本地交互 viewer、完整性
  报告、合成测试和真实左右手 GRAB 验收均已实现。

当前没有实现 TopoRetarget 重定向算法、Arti-MANO/机器人 FK、数值优化、Delaunay/SDF、RL/PPO
或 baselines。阶段 3 是 source-hand adapter，不是机器人重定向，也不声称 MediaPipe detector
accuracy；不进行全量数据集转换。

有界 GRAB 读取器、真实验收命令和误差报告见 [`docs/GRAB_INSPECTION.md`](docs/GRAB_INSPECTION.md)。
这是单条明确选定序列的 60 帧检查，不是全量数据集转换。

## 数据与本地资产

仓库不包含 GRAB、OakInk、OakInk2、ContactPose、TACO、HO-Cap、ARCTIC、DexYCB、MANO 或
SMPL-X。外部数据目录规范为：

```text
<storage-root>/<已登记数据集 alias>/data/**
```

机器相关路径只能写入被 Git 忽略的 `.local/config.yaml`，或通过环境变量设置。可参考
[`configs/paths.example.yaml`](configs/paths.example.yaml) 和 [`.env.example`](.env.example)。

## 常用命令

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

## 开发检查

```bash
ruff check .
ruff format --check .
mypy src
pytest -q
python scripts/check_paper_fidelity.py
git diff --check
```

路线图见 [`docs/ROADMAP.md`](docs/ROADMAP.md)，论文审计见 [`docs/PAPER_FIDELITY.md`](docs/PAPER_FIDELITY.md)，
数据与许可证边界见 [`docs/LICENSE_AND_DATA_POLICY.md`](docs/LICENSE_AND_DATA_POLICY.md)。现有
仓库许可证保存在 [`LICENSE`](LICENSE)；使用论文或本地 Arti-MANO 上游资产时应引用
TopoRetarget 和 ManipTrans。

统一接口见 [`docs/HOI_DATA_INTERFACE.md`](docs/HOI_DATA_INTERFACE.md)，坐标语义见
[`docs/COORDINATE_CONVENTIONS.md`](docs/COORDINATE_CONVENTIONS.md)。
