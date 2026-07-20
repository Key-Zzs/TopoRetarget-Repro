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
不变。在该 Stage 5 snapshot 中，Stage 6 以及后续 geometry、retargeting、collision、SDF 和 PPO
工作尚未开始。

viewer 还实现了只影响显示的 frame stride、播放速度、source/hand/geometry visibility controls，
以及可选 GIF/MP4 无头动画输出。本次审计使用 direct local Zarr store 读写，仍保持标准 Zarr 格式，
以适配受管文件系统；显示操作不改变 canonical schema 或 source arrays。

## Stage 5 semantic contact 与 CLI 收口

contact contract 已根据官方 `otaheri/GRAB/tools/utils.py` 的 `contact_ids` 表收口，来源 commit 为
`4dab3211fae4fc5b8eb6ab86246ccc3a42d8f611`，source SHA-256 为
`bbdae13c1c437d60d22e2e8eabbabb7c2282a47918735876383794739d38a4a7`。tracked mapping 覆盖
`0..55`，保留 `0` 为 no-contact，保留 raw labels，派生 `binary = labels != 0`，并保存官方整数
semantic IDs 和版本化 mapping table。strict 模式遇到未映射 label 直接失败；non-strict 模式使用
明确的 ID `56` 并记录损失。

可视化 CLI 的标准参数是 `--reference-frame`；`--reference` 保留为带 warning 的 deprecated alias，
相同值可同时传入，不同值会显式失败。viewer 支持 source/binary/semantic contact 颜色切换，并在
legend 中显示 mapping identity。当前 closeout reports 与 semantic-enriched real caches 位于被忽略的
`.local/reports/stage5_closeout/` 和 `.local/cache/`。

在显式提供外部 MANO root 后，bounded 真实片段的 fresh MANO-backed semantic conversion 与
validation 已通过。s1 接触窗口报告 `[0,43,46,55]` 且无 unmapped value，raw/binary/semantic/mapping
round-trip 全部精确；s7 双手几何窗口和 table 也通过验证。外部 MANO 文件仍只是运行时输入，未复制到仓库。

## Stage 6 对象几何、确定性采样与 Signed Distance

Stage 6 复用 `MeshDefinition`/`RigidObjectTrack`、现有 SE(3) helper，以及 Stage 4 的
collision-geometry/FK API。只读 mesh audit 记录 source/derived hash、topology、watertight、
winding、degenerate face 和 sign reliability，不修复源数据。

对象点数从 `configs/paper/retarget.yaml` 读取论文锁定值 50。`paper_strict_area_uniform` 使用
按面积加权的 triangle selection、平方根 barycentric coordinate 和显式 NumPy PCG64 seed
`20260720`。保留 face index 与 barycentric，使 scale 变化后仍可精确重建；只在 object frame
采样一次，再逐帧变换，不使用 FPS，也不声明 paper-exact。normals 仅为诊断用途的 face normal。
这些未公开选择登记为 `A_OBJECT_SAMPLING_001`、`A_OBJECT_SAMPLING_METHOD_001`、
`A_OBJECT_SAMPLING_SEED_001`、`A_OBJECT_SAMPLE_TEMPORAL_REUSE_001` 和
`A_SURFACE_NORMAL_MODE_001`。

SDF foundation 使用分块的 analytic point-to-triangle closest point 和 generalized winding
solid angle。`strict` 拒绝 open/non-manifold mesh，`winding` 暴露 confidence/ambiguity，
`unsigned_only` 将 signed distance 标记不可用，不伪造正号。仓库统一约定 positive outside。
Scene query 先使用现有 frame helper 转到 object-local，再把 closest point 和 normal 转回；
edge/vertex 最近点标记 non-smooth，local linearization 只提供几何量，不产生 q-space Jacobian。

机器人采样只使用 Stage 4 `collision_geometry_instances()`。显式 engineering profile 为每个
geometry 32 点，不是论文值。RH/LH Arti-MANO 每侧均有 16 个 collision geometry 和 512 个点；
只有 visual 的 tip link 会报告，不会补造 collision。pointwise collision probe 包含
link/geometry/sample identity、sign confidence 和 penetration depth，但不构造最终 `Q_t`、
Delaunay、Laplacian、slack 或优化。

有界验收使用 `s7/cubemedium_inspect_1` 的 `[0,60)` 帧和本地 RH/LH 资产。报告/图片位于被忽略的
`.local/reports/stage6/`，derived sample cache 位于 `.local/cache/geometry/`。source NPZ、mesh、
canonical cache、MANO 和 Arti-MANO asset hash 未变化。在这个 Stage 8 之前的 snapshot 中，Stage 7
已完成并保留显式假设；后续 Stage 8 closeout 见下文。

## Stage 7：相对骨方向 Warm Start（2026-07-20）

编辑前先验证 Stage 6 closeout：提交为 `8c5b1c7`，index/worktree clean，中英文 Stage 6 文档已
完成。Stage 7 复用现有 canonical `mediapipe21` scene track、Stage 4 可微 Arti-MANO FK/anchors，
并写入独立的 warm-start Zarr artifact。优化不读取 Stage 6 object samples、SDF、Delaunay、
Laplacian、碰撞查询或 PPO 模块。

默认 frame profile 是 `canonical_keypoint_wrist_v1`：wrist 原点、middle-MCP longitudinal 轴、
经过 Gram-Schmidt 的 index-minus-pinky lateral 轴，以及叉乘得到的第三轴由 source/robot 共同使用。
translation-centered scene-axis profile 作为有界 observability 对比保留。两个 profile 的 strict
模式均显式拒绝退化帧，并保留 RH/LH 语义顺序；不会把 GRAB stored wrist pose 静默当作 Arti-MANO
palm frame。

默认 bone profile 是五条完整手指链、20 根 directed bones、15 个同手指相邻 pair；phalange-only
诊断 profile 是 15 根骨和 10 个 pair。单位方向和不再次归一化的相邻差分支持 batch 与 autograd。
Eq. (1) 使用精确 sum，不是 mean、夹角 loss、绝对方向 loss 或骨长加权 loss。

Eq. (2) 使用 raw 22-joint radians；`lambda_warm=1`、`lambda_smooth=2.5` 从 paper config 读取。
第一帧用 neutral q 且不使用 temporal residual，后续帧只使用上一帧成功的 warm-start q。solver
是 float64 Torch-autograd Jacobian + SciPy TRF，并直接使用 URDF bounds。artifact 自行计算 paper
objective，不把 SciPy 的 half-cost 当作论文目标；strict 失败会报告 frame/status 并终止。

local direction objective 使 base translation 不可观，默认 local profile 同时消除了 base rotation
可观性。因此 solver 只优化 q_theta，同时记录 qpos Jacobian singular values/rank 和 synthetic base
Jacobians。求解后使用显式的、非论文已公开的
`T^S_B=T^S_Hs(T^B_Hr(q))^-1` 生成 base seed，并将对齐误差写入 artifact 和 validation report。

有界真实验收使用 `s7/cubemedium_inspect_1`、`[0,60)`、native 120 FPS 和本地 Arti-MANO RH/LH 资产。
RH 使用 `cubemedium_inspect_1_rh_f000000_f000060_mp21.zarr`；LH 使用 Stage 5 已存在的
`semantic_left_f000000_f000060.zarr`。两侧均生成 60 帧、22-D qpos 的
`toporetarget.warm_start.v1`，solver 全部成功、逐帧 total objective 不增加、FK/base frame 对齐
通过、source cache hash 匹配。artifact 和报告/图片位于被忽略的 `.local/`。

新增测试覆盖 20/15 与 15/10 topology、RH/LH frame 语义、刚体不变性、translation-centered 诊断、
strict 退化、精确 Eq. (1) sum、Torch float32/float64 autograd、Eq. (2) residual scaling、base
不可观和 artifact round-trip。Stage 7 状态是 `implemented_with_assumptions`；Stage 8 随后完成有界 closeout。

## Stage 8：Source Interaction Graph 与 Laplacian Loss（2026-07-20）

Stage 8 新增独立的 source-only interaction graph artifact 和冻结的 Eq. (7) evaluation
artifact。RH/LH 两条 bounded clip 各使用 60 帧、21 个 canonical MediaPipe-21 source 点和
固定的 Stage 6 50 点 object sample artifact。每个 source frame 只运行一次显式的
non-incremental SciPy/Qhull Delaunay（`Qbb Qc Qz Q12`），随后将完整唯一 tetrahedron edge
和 source-derived directed weight 复用于 robot FK。strict profile 仅在 Qhull 输入上使用
centroid/bounding-box-diagonal conditioning；source vertex、volume、distance 和 weight 仍
全部保留米制 scene frame。

Eq. (6) 使用可微 Torch sparse scatter Laplacian，Eq. (7) 严格按 71 个 vertex 的 mean squared
residual 计算。evaluation 只读取 Stage 7 qpos/base，不修改它们，保留 object point identity，
输出 qpos Jacobian 和有界 base diagnostic，并记录 robot-side Delaunay、optimization、SDF
和 collision access 均为 0/false。Eq. (8)-(9)、slack、碰撞约束和 RL 仍未实现。

RH/LH graph/evaluation validation、identity/scaled-residual oracle、topology-over-time、
object-scale diagnostic、input/source-integrity audit、unit tests 以及静态和交互式可视化
smoke test 均写入被忽略的 `.local/`。Stage 8 状态为 `implemented_with_assumptions`；没有修改
Stage 6/7 artifact 或 source hash，也没有执行 git commit/push/tag。

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
