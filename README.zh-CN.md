# TopoRetarget-Repro

[English README](README.md)

TopoRetarget-Repro 是 TopoRetarget 论文的非官方、独立、可追踪复现仓库基础。初始目标是
GRAB → Arti-MANO，后续目标是支持多种 HOI 数据集以及任意 URDF/MJCF 灵巧手。

## 当前状态

- 阶段 0 complete：仓库脚手架、配置、只读数据集发现和本地 Arti-MANO 导入器。
- 阶段 1 complete：完整 16 页论文审计、参数来源、assumption 和忠实度检查器。
- 阶段 2A complete：统一 HOI schema、明确坐标语义、可选 Zarr 缓存、确定性合成数据、误差
  指标和无头比较可视化已实现。
- 阶段 2B not started：真实 GRAB 检查读取器是下一步的有界任务。

当前没有实现 TopoRetarget 重定向算法、机器人接口、MANO 到 MediaPipe 映射、数值优化、
Delaunay/SDF、RL/PPO 或 baselines。阶段 2A 不进行全量数据集转换，也不依赖机器人模型。

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
