# 开发日志

## 2026-07-27 -- F0 tracked 机器人手资产与通用 target-hand foundation

在 `main` 完成 F0。Arti-MANO 现在是仓库内 tracked vendor snapshot，路径为
`third_party/robot_hands/artimano/`，固定到 ManipTrans commit
`a3d08cfe3c3a5868a7f057533bcaf759c5af4705`。snapshot 含两个 URDF、96 个 mesh 文件以及
`LICENSE`、`SOURCE.yaml`、`NOTICE.md` 和 hash manifest；没有复制 ManipTrans Python 源码。
source manifest hash 为
`1d14cce93e2ee09dedbfcda842b1d8aac29443f86b57a0a15f6289bd55e0f771`。

通用 target-hand contract 新增 asset bundle、kinematic、semantic-anchor、surface、collision 和
simulation spec。Arti-MANO RH/LH 统一由 `RobotHandRegistry` 构造，旧的
`load_artimano_model` API 继续保留并委托 registry。resolver 会记录 tracked、显式 override 或
deprecated legacy fallback；新增 `robots resolve-assets` 与 `compare-assets` 命令展示 provenance
和迁移状态。anchor/Jacobian shape 改为由 profile 驱动，不再假定所有手都是 21 点或 22 DoF。

F0 audit 对两侧 tracked 与原有本地 payload 比较了 topology、qpos 顺序、limits、FK、anchors、
Jacobians、mesh transforms 和 source bytes，结果均 exact。Stage 7–10 历史报告/导出保持只读，
通过 source hash 重绑定；没有调用 solver，也没有重写历史 artifact。证据位于 `.local/reports/f0/`
且保持 untracked。Wuji Hand2、`develop/pene-loss`、SDF penetration loss、solver profile 修改和
RL 均不属于 F0。

## 2026-07-27 -- W0/W1 Wuji Hand2 Beta1 通用 target 集成

在 `main` 完成 W0/W1。批准的 `wuji-technology/wuji-description` Hand2 Beta1 body 子集已跟踪到
`third_party/robot_hands/wuji_hand2_beta1/`。请求的 `release/v2026.7.23` 解析到 commit
`2b57d2621caed4e65207bb767ba25fc8eaec0881`；同版本号 tag 指向不同 commit，没有替换。`SOURCE.yaml`
记录 MIT license、排除路径、import tool version、per-file hash 和 source manifest hash。

增加独立的左右手 generic spec：26 links、25 joints、20 actuated joints、5 fixed joints，root 为
`r_wrist`/`l_wrist`，显式 20-DoF qpos/URDF/MJCF/actuator order、MediaPipe-21 anchors、surface
policy 及独立 URDF/MJCF collision profile。不依赖 MuJoCo 的 consistency check 在 neutral、midpoint
和随机 qpos 上通过，覆盖 axes、limits、root、tip sites、link transform、anchors、mesh reference
以及十个 MJCF contact-exclude pair。soft-pad tip mesh 保留为 visual payload，不静默加入 formal collision。

Stage 7/8 通用路径已移除历史 22-DoF 假设。airplane `[240,243)` 有界窗口通过 warm-start、
source-only graph/evaluation、collision QuerySet 和 Stage 9 objective/constraint/Jacobian construction
smoke；报告明确 `optimization_performed=false`，不声称 W2 的完整多 clip Wuji 重定向。
`develop/pene-loss` worktree、上游 checkout、canonical source、object samples 和历史 artifact 均未修改。

## 2026-07-24 -- GRAB Arti-MANO 质量 A–E

实现固定的 G1–G4 质量实验。阶段 A 冻结原生帧范围并验证 right-hand identity，复用匹配的
airplane artifact，并通过 checkpoint/resume 运行保留的 v2/v3 solver profile。阶段 B 增加
通用 visual surface contact contract 与确定性的 Arti-MANO region。阶段 C 增加 seed-only
morphology candidate 和固定归一化 prior diagnostic。阶段 D 声明完整 contact grid，并在不
削弱 collision constraint 的前提下记录 proxy gate。阶段 E 记录 E0–E3 lineage、Pareto gate、
baseline fallback、报告和四个自包含 HTML viewer。

所有新输出只在 `.local/experiments/grab_artimano_quality_v1/`；raw source 与历史 Stage 5–10
artifact 保持只读，`.local` 不进入 Git。ContactPose 保持 deferred，结论仅限 `s1` 的
within-subject multi-object development benchmark。

第一次真实阶段 A 执行发现了通用的 active-set continuation 舍入问题：reference/solver SDF
转换后，新 soft constraint 的初值可能约有 `2.7e-8 m` 的微小不可行量。已加入固定的
`1e-9 m`、仅作用于初始化的 interior buffer；objective、solver YAML、tolerance、QuerySet、
active margin 和 acceptance gate 均未改变。修复后 G2 的两个保留 profile 都完成了 `60/60`
strict-accepted frame，并生成有效 full-512 artifact。随后按授权的
`hybrid_original_distance_proxy_sign_v1` contract 处理 open banana mesh：原始 distance 和
visual/contact 语义仍来自 source mesh，derived proxy 只用于 sign。proxy 通过固定 geometry
gates，但首个 active QuerySet 仍有 3 个 sample 位于原始 boundary exclusion zone，其中 2 个
落在 synthetic patch face。执行因此在 G3 以
`hard_blocker=SIGN_PROXY_CONTACT_REGION_CONFLICT` fail-closed；没有修改 margin、frame、profile、
raw mesh，也没有继续 G4/C–E。记录状态为 `GRAB_QUALITY_A_TO_E_BLOCKED`。

## Q1–Q3 多数据集交互 benchmark（2026-07-24）

新增版本化 `toporetarget.hoi_benchmark.v1` selection contract、lazy GRAB selector、
ContactPose 目录/schema adapter、分离的 metric registry、Eq. (10)–(12) 实现、GRAB contact
proxy、动态/静态适用性、绑定 manifest 的 baseline 执行、自动 gate、macro aggregation 和
self-contained HTML dashboard。baseline 执行前先写入并锁定 selection；原始数据和 official
artifact 只读。保留固定的 `s1/airplane_lift`、right、global `[240,300)` unit。本阶段不修改
Eq. (1)–(9)，不做 per-unit tuning，也不宣称 Eq. (9) 的 author-exact 语义。

本 checkout 的只读执行证据：GRAB/ContactPose/MANO/Arti-MANO required paths 和显式
`topo-retarget` imports 通过 preflight。现有 GRAB index 有 1,334 条 non-fixed entry；有界
16-entry native-contact probe 选择了 `s1/apple_eat_1 [212,272)`、`s1/banana_lift [1658,1718)`、
`s1/alarmclock_lift [407,467)`，并保留 fixed `s1/airplane_lift [240,300)`。ContactPose 检查
得到 110 条 annotation candidate、0 条 selected，110 条被记录为
`official_contact_annotation_unavailable_or_unrecognized`（其中 12 条同时属于 diagnostic
deep-concave exclusion）。状态为 `Q1_CONTACTPOSE_SELECTION_BLOCKED`；selection 未 freeze，未运行
baseline/evaluation。

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

## Stage 9：最终保交互约束 refinement（2026-07-20）

Stage 9 在现有 RH/LH `s7/cubemedium_inspect_1` 60 帧输入上实现有界的 Eq. (8)-(9)。solver
读取冻结的 Stage 7 warm start、Stage 8 graph、Stage 6 cubemedium mesh 和不可变的 512 点
robot collision surface。实现使用显式 local seed-delta 坐标、previous-final 到当前 seed 的
时序 remap、`configs/paper/retarget.yaml` 中的 paper weights、positive-outside reference SDF、
每个 query 的 slack，以及独立 full-surface audit。

Full/adaptive QuerySet profile 是确定性的并写入 hash。adaptive profile 从 penetration、10 mm
margin 和每个 geometry 最近点开始，并单调加入 full-surface violation。solver 是 float64 SLSQP，
目标函数使用 Torch autograd，SDF constraint Jacobian 使用 hybrid 策略。convex-hull solver backend
只有在与 Stage 6 reference backend probe 对比通过后才使用；最终验收始终使用 reference backend。

本阶段修复了零角度 SO(3) 梯度 NaN、geometry sample slice 错误、solver/reference SDF 混用和
async Zarr array 创建/读取问题。engineering profile 记录 `maxiter=30`、`ftol=1e-7` 和 fail-fast，
因为论文没有公开 optimizer 细节。Stage 9 仍是 `implemented_with_assumptions`；Stage 10、RL、
physics、ContactPose 和 baseline 未启动。所有输出和 pre-stage snapshot 留在被忽略的 `.local/`，
没有执行 git add/commit/push。

最终 closeout 对 RH/LH 都重新运行了完整的 `[0,60)`。RH 的最小 full signed distance 为
`0.623582905 m`、penetration 为 0、最大 slack 为 `2.137e-6 m`，平均/p95 solve time 为
`20.146/22.435 s`；LH 分别为 `0.641271031 m`、0、`5.096e-7 m` 和 `19.214/20.853 s`。
两侧 independent validation 都通过；每侧 original 与 full rerun artifact 的 47 个非时间数组
都完全一致。`0/29/59` 三帧中 adaptive 使用 16 个 query，full reference 使用 512 个 query；
最大差异为最小 full SDF `8.20e-6 m`、objective `8.77e-10`。RH/LH Jacobian 检查通过，
constraint 最大误差低于 `2.03e-10` 且没有 finite-difference fallback。报告位于
`.local/reports/stage9/`；已知的 canonical `metadata.json` Zarr sidecar warning 是既有输入
侧车，不会修改 source artifact。

## 阶段 10：有界 workflow 编排（2026-07-20）

阶段 10 增加 manifest 驱动的 19 节点 GRAB→Arti-MANO DAG、官方 semantic contact-window 选择、
内容签名 cache/resume/invalidation、raw source integrity、semantic sanity 与跨阶段 identity 报告、
artifact-only review，以及只读 `robot_reference.v1` 导出。配置在 `configs/workflows/`，流程文档在
`docs/END_TO_END_GRAB_ARTIMANO.md`、`docs/WORKFLOW_RESUME_AND_PROVENANCE.md` 和
`docs/TRAJECTORY_VISUALIZATION.md`。

选择器在 `s1/airplane_lift` 的 `[844,904)`、`[240,300)`、规格指定的 `[238,298)`，以及已有
Stage 9 对象 `s7/cubemedium_inspect_1` 的 `[363,423)` 窗口通过官方右手 semantic contact 和严格
watertight mesh 门禁。额外的有限显式候选包括 `s1/airplane_fly_1 [729,789)`、
`s1/cubemedium_inspect_1 [343,403)` 和 ratio=0.5 的 transition window
`s1/airplane_fly_1 [159,219)`。`s1/cubesmall_inspect_1 [984,1044)` 虽通过 selector/mesh，
但未修改的 Stage 8 strict graph 在第 13 帧因两个 simplex volume 低于 tolerance 被拒绝。其余
已完成的 contact-rich run 都到达 frozen interaction evaluation，随后在未修改的 Stage 9 SLSQP
refinement 返回 `Iteration limit reached`（第 0 或第 1 帧）。transition run 的 bounded runtime
异常，solver 子进程超过 40 分钟且 CPU 约 100%，因此被停止并记录为 SIGTERM failure，未伪装成成功。
没有修改 Stage 7–9 solver、weight、coordinate 或 threshold；Stage 10 编排已实现，但真实验收在
contact-rich refinement 收敛边界处 blocked。每个 run 的 input、reuse、performance、待完成的
determinism、semantic、source-integrity 和 summary 报告保存在被忽略的 `.local/runs/stage10/`，
没有执行 commit、tag 或 push。

之后对明确列出的 `s1/apple_lift`、`s1/cylinderlarge_inspect_1`、
`s1/spheremedium_inspect_1`、`s1/mug_lift`、`s1/phone_lift` 和
`s1/stanfordbunny_inspect_1` 做了有限 selector 查询。Apple 和 cylinderlarge 通过严格选择；
sphere 在未修改的 Stage 8 graph 验证处失败，mug、phone 和 stanfordbunny 因 mesh 非 watertight
被拒绝。新的 `cylinderlarge_inspect_1 [327,387)` 仍通过 Stage 8，随后在 Stage 9 第 0 帧返回
`Iteration limit reached`。对 `airplane_lift [240,300)` 做的只读单帧诊断记录了冻结
`maxiter=30` 下的 SLSQP status 9，但返回候选的 full-surface 最小 signed distance 已为
`+0.01184 m`，hard/soft residual 也为正；这确认既有 strict fail-fast solver 边界，Stage 10
不放宽它。诊断保留在 `.local/reports/stage10/contact_rich_solver_diagnostic.json`。

对应的左手有限查询 `s7/cubemedium_inspect_1 [513,573)` 通过 contact 和严格 mesh 选择，
但在未修改的 Stage 8 graph 第 1 帧因一个 simplex volume 小于等于 `1e-24` 失败，未进入
final refinement。

## Stage 9.1 solver-robustness closeout（2026-07-21）

Stage 9.1 保留 v1 SLSQP profile，并新增独立的 contact-rich v2 profile。原有 active-set
问题是在 QuerySet 扩展后重新使用 Stage 7 warm seed 初始化。v2 改为从上一轮
`result.x` continuation：直接复制 base/q，按 query ID 映射旧 slack，新 query 使用有界
最小可行 slack 公式。query set 只允许单调增长，continuation trace 写入 artifact provenance。

现在把 optimizer status/counter 与 primal、bounds、active-set、full-surface hard/soft、
finite-value 和 acceptance 字段分开保存。即使 status 9 的候选可行，strict policy 仍拒绝；
`feasible_stationary_v1` 只登记为 deferred，未启用。固定 benchmark grid 与 deterministic
repeat 的权威记录在 `.local/reports/stage9_1/maxiter_benchmark.json`：`[30, 60, 100, 200, 400]`
共 35 条记录，最小统一预算为 `100`。保留的 v1 profile hash 为
`6affff2fdb425a0402f643c291c0b8904d4dbec6c5b69a5006cf9829dcc220aa`，v2 profile hash 为
`c42c21d894c54d07b1d30943b5a3338b13628bf0429ab203b5540cf934d09b7c`。
固定 benchmark 在 100 通过，但 bounded runtime 内尚未产出 opt-in 的完整 60 帧 contact-rich
artifact 及其 deterministic repeat，因此 Stage 10 仍保持 blocked，等待该真实 run。窗口几何
及 far-vs-contact 对照记录在 `.local/reports/stage9_solver_closeout/`；这不是放宽 status-9
acceptance。Stage 10 resume 显式选择 v2，只使 Stage 9 及下游 signature 失效，并复用 Stage 5–8
artifact。solver 与 termination 仍是论文未公开的实现假设。
随后为 Stage 9.2 性能与可恢复执行阶段暂停了 bounded v2 重跑；完整序列仍被性能阻塞。
因此本次只保留已测试的 closeout 修改，不宣称 Stage 9.1 complete，也不宣称已经生成
60 帧 artifact 或 deterministic repeat。

## Stage 9.2 refinement 性能与可恢复执行（2026-07-21）

Stage 9.2 在冻结的 Stage 9 数学契约外增加执行层：不可变 per-frame context、精确
float64 x/query cache 失效、持久 mesh/SDF resource、批量 collision-point Jacobian、
明确的 full-512 audit 调度、strict-accepted frame 的原子 checkpoint、soft wall-time
pause、resume、assembly 和 fresh/resumed comparison。solver profile、paper weights、
signed-distance convention、512 samples、v2 continuation 及 strict status-9 rejection
均未改变。

CLI 新增 `profile-refinement`、`checkpoint-status`、`validate-checkpoints`、
`assemble-refinement` 和 `compare-refinement-runs`。execution profile 使用独立的
CPU float64 `cached_checkpoint_cpu_float64_v1`。Focused tests 已通过；完整 contact-rich
60 帧运行、deterministic fresh/resumed repeat 和 runtime gate 仍需真实证据。在这些
report 出现前不解除 Stage 10。详见 [`REFINEMENT_PERFORMANCE.md`](REFINEMENT_PERFORMANCE.md)、

## Stage 10 已接受的 reference-runtime milestone（2026-07-21）

用户已接受 `s1/airplane_lift` 右手 `[240,300)` 窗口。新的 manifest 位于
`.local/runs/stage10_reference_runtime/`，只引用已接受的 Stage 9.2 final artifact；
Stage 5–9 全部记录为 reused，Stage 9 solver invocation 为 0。跨阶段 identity、
semantic sanity、NPZ/direct-Zarr round-trip 和静态 viewer smoke 均通过；preferred
performance gate 仍为 false，performance debt 仍保持 open。
这个已接受的 bounded reference-runtime milestone 覆盖早期 pre-v3 Stage 10 blocked-run
summary；早期条目仍保留，作为旧 contact-rich window 的历史失败证据。
[`REFINEMENT_CHECKPOINT_AND_RESUME.md`](REFINEMENT_CHECKPOINT_AND_RESUME.md) 和
[`stages/STAGE_9_2_REFINEMENT_PERFORMANCE.md`](stages/STAGE_9_2_REFINEMENT_PERFORMANCE.md)。

随后 bounded full run 完成全部 60 个 contact-rich frame，solver compute 为
`1075.941 s`（`17.932 min`）。这份较早的 v1 证据已由下方 v3 优化 run
取代，但其 strict acceptance 与恢复历史仍保留在 report 中。

v3 execution profile 使用 analytic URDF spatial Jacobian、status-9 的 strict
reference recovery，以及 leaf size 为 512 的 SDF tree。第一次运行完成 60/60 帧，
median 为 `10.766 s`、p95 为 `38.711 s`、总 solve time 为 `1104.827 s`；deterministic
repeat 完成 60/60 帧，median 为 `10.773 s`、p95 为 `39.052 s`、总计 `1107.368 s`。
两次运行均为 status 0 且 strict accepted，checkpoint chain 有效，独立 `60 x 512`
reference validation 通过，最大 signed-distance error 为 `2.50e-16 m`。排除
`solve_time_s` 与文档规定的 metadata 后，全部持久化数组 exact equal。最终状态仍为
`STAGE9_2_COMPLETE_REFERENCE_RUNTIME`；preferred single-frame gate 未通过，但用户明确
接受的 reference-runtime Stage 10 bounded milestone 已完成。

## Stage 10.x interaction-mesh HTML 可视化（2026-07-21）

在 manifest 驱动的 mesh viewer 中增加了五个可切换模式：`mesh`、`full-graph`、
`figure4-style`、`laplacian-diagnostic` 和 `combined`；所有模式都保留 source/warm-start/final mesh。页面直接复用已接受的 Stage 8
graph/evaluation artifact，保持 21 个 hand 点 + 50 个 object 点的 vertex contract 和冻结的
有向权重，并仅在内存中计算 final Laplacian residual 供诊断使用。新增 edge category、
threshold/top-k、residual target/scope、scalar/vector、labels 和 state-layer 控制；不运行
solver、不重建 graph，也不写入输入 artifact。

已对接受的 `s1/airplane_lift` 右手 `[240,300)` run 生成五种 HTML，并通过 headless Chrome
smoke。生成前后 artifact content hash 与 mtime 均未变化。该 viewer 只是检查工具；正式的
interaction、collision、continuity、bounds、solver 和 provenance gate 仍以数值报告为准。
本阶段没有执行 git add、commit 或 push。

## Stage 9.3 接触保持与碰撞几何审计（2026-07-22）

实现了 manifest 驱动的审计 workflow 与 self-contained HTML review，并对已接受的
`s1/airplane_lift` 右手 `[240,300)` reference runtime 完成完整 60 帧运行。审计使用
确定性的 dense surface sample、明确的 positive-outside signed-distance provenance、
source/warm/final contact proxy、visual-vs-collision offset、QuerySet per-point/per-link
报告、同定义 Stage 9 objective 对比和不执行优化的 warm-to-final interpolation 诊断。
所有正式输入的 hash 与 mtime 均保持不变，solver invocation count 为 0。由于现有
artifact 不支持更强结论，contact retention 与 physical trackability 仍分别保持
diagnostic/inconclusive 或 unverified。可选 shadow evidence 未运行，并明确记录为缺失。
没有执行 git add、commit、push、reset 或 tag。

## Stage 9.3.1 signed-distance reconciliation 与 bounded shadow gate（2026-07-22）

新增只读 reconciliation workflow 和 CLI，并针对已接受的 `s1/airplane_lift` 右手
`[240,300)` reference runtime 完成验证。结果证明 Stage 9.2 持久化 512 点的 identity/order
与 transform chain；独立 acceptance replay 为 `60/60` 且 0 mismatch，并记录 signed-distance
definition matrix。持久化值与统一的 reference triangle/winding backend 在 machine precision
内一致。旧 Stage 9.3 的 `convex_hull_exact_solver_only` 值与该定义不一致，因此唯一状态为
`RETURN_TO_STAGE9_2_ACCEPTANCE_OR_METRIC_FIX`。

由于 visual mesh 是 open surface，directional offset audit 不会把原有 unsigned-only
inflation label 继续当作已证实结论，而是保持 inconclusive。shadow boundary 选择了 3 个
frame，但因 reconciliation gate 失败没有运行 profile，solver invocation 为 0。正式 Stage
9.2/Stage 10 artifact、export 和 manual acceptance 均未改变；所有修改仍未 staged。

## Stage 9.3.2 canonical SDF 重审计（2026-07-22）

新增版本化 `reference_winding_v1` 正式 contact-distance contract、v2
canonical re-audit workflow、legacy-vs-canonical 差异报告、source/warm/final
proxy 指标、collision/visual coverage 诊断、readiness 报告和隔离的六种
bounded shadow 边界。正式 evaluation 与已批准的 Stage 9.2 solver backend
分离；legacy convex-hull 结果仅用于诊断并对正式 contact claim 标记为
superseded。workflow 保留 Eq. (1)-(9)、论文权重、所有已接受的 Stage
9.2/Stage 10 artifact、manual acceptance 和 robot export。audit-only solver
invocation 保持为 0；若 gate 通过，shadow 输出仍是隔离的 diagnostic。所有
代码修改保持 unstaged。

## Stage 7.1 Warm-Start 保真度与可达性审计（2026-07-23）

新增 manifest 驱动的只读 `workflow audit-warm-start` boundary 和 HTML/report 契约。对已接受的
`s1/airplane_lift` RH reference runtime（local `[0,60)`、global `[240,300)`）重放全部 persisted
Stage 7 qpos；Eq. (1)/(2)、frame、base、source mapping 和 robot mapping gates 均在 machine precision
内通过。审计记录 thumb URDF ancestry/axis、joint-limit margins、Jacobian projection、Kabsch
alignment alternatives、raw 与 robot-length thumb target，以及 warm/final formal/keypoint/E_IM/contact
attribution。

5 个 bounded frame 执行 diagnostic-only IK 和 4096 点 Sobol workspace sampling。raw thumb target
到 sampled workspace 平均约 12.51 mm，而 robot-length target 平均约 3.81 mm，且每个 selected frame
均接近 workspace；该结论记录为 morphology/embodiment gap，不修改正式 Stage 7 目标。whole-hand
final keypoint RMSE 和 E_IM 高于 warm，因此 final refinement degradation 作为独立 ranked factor。
最终 readiness 为 `WARM_START_FORMALLY_VALID_CONTINUE_STAGE9_3_3`，
`CONTINUE_STAGE9_3_3=YES`；official solver invocation 仍为 0，45 次 diagnostic call 全部隔离。没有
修改正式 artifact、Stage 10 manifest、manual acceptance 或 Git index。

## Stage 9.3.3 shadow 等价性与长指归因（2026-07-23）

新增隔离的 `shadow_equivalence.py` workflow、CLI、基于 repeat 的版本化数值
contract、context binding、profile isolation、原子 checkpoint/resume、逐指/反事实/
gradient/constraint 报告和 HTML boundary。对 3 个 selected frame 各做 3 次 official
baseline replay；QuerySet IDs/order 与 strict feasibility 相同，但 qpos 和 canonical
SDF 差异远超预先声明的 cap。Stage 10 manifest 内部 commit provenance 也不一致
（`23e6465` 对 `58fa77c`）。结果为：
`SHADOW_BASELINE_NOT_NUMERICALLY_EQUIVALENT`、
`RETURN_TO_STAGE9_3_2_SHADOW_HARNESS_FIX`、`ENTER_STAGE9_4=NO`；mandatory
shadow profile 运行数为 0。正式 artifact 和 Git index 未改变，代码仍未 staged。

## Stage 9.3.4 provenance-rebased 因果实验（2026-07-23）

新增隔离的 provenance audit、detached historical lane contract、current-lineage
baseline、有界 multistart/base-seed 诊断、QuerySet/margin mandatory profile、保守
readiness report 和 HTML handoff。historical exact environment 不可用时 fail-closed；正式
artifact、Stage 10 manifest、manual acceptance 和 Git index 保持不变，新增输出均为诊断且未 staged。

current-lineage baseline 的 60 帧全部通过：status 0、strict acceptance、连续 checkpoint
chain、finite full-512 audit，且 raw penetration 为 0。由于记录的 package environment 与当前
环境不完全匹配，historical exact replay 保持 unavailable。5 个 selected frame 完成 60 个
multistart variant、60 个 base-seed final variant 和 15 个 formal mandatory profile variant；
projection 行明确标记为未求解 diagnostic。最终路由为 `STAGE9_3_4_INCONCLUSIVE`，
`ENTER_STAGE9_4=NO`，等待人工决策。正式 artifact 和 Git index 未改变。

## Stage 9.3.5 可行性投影与因果闭环（2026-07-23）

新增隔离的 Stage 9.3.5 可行性路径扫描、canonical full-512 状态投影、反事实状态、
目标与约束归因、分支 rollout gate、报告、HTML handoff、测试和同步文档。该 workflow
仅用于 diagnostic，保持正式 Stage 9.2 solver contract、Stage 10 artifact 和 Git index
不变。

本次有界运行完成了 frame-10 的 1001-sample canonical full-512 path scan，以及只读归因
bundle：12 个 counterfactual state、6 条 objective row 和 512 条 constraint-pressure
row。由于 CPU full-512 constraint evaluation 在 wall-time 内未完成，projection attempt
在有效 solver checkpoint 前暂停；没有任何 projection result 被当作 accepted。最终路由为
`RETURN_TO_PROJECTION_DIAGNOSTIC_HARNESS_FIX`，`ENTER_STAGE9_4=NO`。

## Stage 9.3.5 continuation 与五帧闭环（2026-07-23）

在不修改 canonical evaluator 的前提下修复了 diagnostic harness：resume 现在会验证并复用
full-512 path cache；projection callback 复用已经 cross-validation 通过的
`convex_hull_exact_solver_only` solver-side backend，而每个 candidate 仍由 canonical
`reference_triangle_winding` 独立复核。minimal profile 同时修复了 zero-slack Jacobian
padding 问题。所有修改仍为 diagnostic 且保持 unstaged。

请求的 selected frames `[10,39,30,36,0]` 现在均有 1001-sample、512-sample path cache
和 compact feasibility report。两个 projection profile 均在五个 frame 上运行（共 10 次
attempt）；其中 3 次为 strict accepted projection，其余明确记录为 `status=9` solver
failure，绝不把未收敛结果写成 accepted。独立 canonical validation 对报告的 candidate
完成 finite/full-512、bounds、hard/soft、slack 与 raw-penetration 检查。

完整只读因果 bundle 包含 90 个 counterfactual state、90 条 objective attribution row
和 2,560 条 constraint-pressure row。bounded branch gate 正确返回
`NOT_REQUIRED_BY_GATE`，因为没有 candidate 满足 improvement gate。最终路由为
`READY_FOR_STAGE9_4_REFINEMENT_ENGINEERING_REPAIR`，并明确
`ENTER_STAGE9_4=NO`、`HUMAN_DECISION_REQUIRED=YES`、
`STOP_AFTER_STAGE9_3_5=TRUE`。正式 Stage 7/8/9/10 artifact、current-lineage
baseline、manual acceptance 和 Git index 均保持不变。

## Stage 9.3.5 因果门槛修正（2026-07-23）

最终报告审计发现，上述初始 aggregate route 过于宽松：仅有较低的 state
fraction 不能证明 `OFFICIAL_FINAL_MOVES_BEYOND_FEASIBILITY`。声明的 gate
还要求至少 2 个 representative frame，long-finger RMSE 至少改善
`max(1.0 mm, 10%)`，并同时满足 strict canonical feasibility 与 projection
更接近 warm。刷新后的 checkpoint validation 显示 2 个 distinct frame 上有
3 次 strict accepted attempt，但所有报告的 long-finger improvement 均为负，
因此没有 frame 通过完整 causal gate。

当前报告将 frame 39、30、36、0 分类为 `WARM_ALREADY_FEASIBLE`，frame 10
分类为 `INCONCLUSIVE`，aggregate route 修正为
`STAGE9_4_NOT_YET_JUSTIFIED`。明确
`ENTER_STAGE9_4=NO`、`HUMAN_DECISION_REQUIRED=YES`、
`STOP_AFTER_STAGE9_3_5=TRUE`。没有修改正式 artifact、current-lineage
baseline、Stage 10 artifact、manual acceptance 或 Git index。

## 2026-07-24 -- Stage 9 一次性因果闭环

完成了有界 projection contract、Eq. (9) implementation map、固定 C0--C7
selected-frame sweep、单一 root cause 决策以及唯一允许的 faithful
regularization repair。Projection 仍仅用于诊断；完整 60 帧验证和人工 review
状态记录在 `.local/reports/stage9_one_shot/`。

## 2026-07-24 —— G3 派生 sign proxy 与严格冲突审计

新增确定性的 derived watertight sign-proxy 流程和
original-distance/proxy-sign hybrid backend。banana 原始网格保持只读；
Candidate 1 local repair 通过固定 20k surface gates，并保存 boundary loops、
near-zero IDs、patch IDs、hash 和 provenance。G1/G2/G4 的 identity validation
通过。

恢复 A–E 后，G3 Stage A 到达正式 gate，随后正确 fail-closed 为
`SIGN_PROXY_CONTACT_REGION_CONFLICT`：3 个 active QuerySet sample 位于原始
boundary exclusion zone 内，其中 2 个最近 proxy face 是 synthetic patch。
没有修改轨迹、margin、frame、原始资产或历史 Stage 10 artifact。G3 geometry
audit HTML 及 first/middle/last PNG 已生成；G3 下游 C–E 和总 A–E 推荐结果仍
无效。
