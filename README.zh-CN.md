# TopoRetarget-Repro

[English README](README.md)

TopoRetarget-Repro 是一个非官方、独立、可追踪的
[*TopoRetarget: Interaction-Preserving Retargeting for Dexterous Manipulation*](https://arxiv.org/abs/2606.16272)
复现仓库。项目提供与机器人无关的 HOI 数据接口、明确的坐标约定、source-hand 转换工具、
复现审计，以及面向完整灵巧手重定向的分阶段路线。

当前实现已经覆盖 canonical HOI interface、有界 MANO→MediaPipe 风格 21 点 source-hand
adapter、Stage 4 通用机器人手/Arti-MANO 目标运动学接口、有界的 Stage 5 GRAB 数据集
adapter、Stage 8 source-only interaction graph/Laplacian loss，以及有界的 Stage 9
Eq. (8)-(9) final refinement；仍未声称实现 RL pipeline 或论文实验结果。

## 仓库概览

主要入口是 toporetarget CLI。仓库按完整功能组织：

- 统一的、与机器人无关的 HOISequence 数据结构、scene-frame 几何和显式 SE(3) 变换；
- 对单条 GRAB NPZ 的只读检查，以及到 canonical Zarr 的转换；
- 显式 MANO 语义 layout 和版本化 MANO→MediaPipe21 mapping profile；
- 通用可微 URDF 手部 FK、命名 qpos、目标锚点和 Arti-MANO 左右手检查；
- lazy GRAB index、保留原生时间/网格的单序列 adapter、contact modes、validation、provenance
  和 raw/canonical 对比；
- source/object/timestamp 保留报告，以及静态和本地交互式几何 viewer；
- 论文忠实度审计、assumption 记录和 tracked Arti-MANO 资产 provenance 支持。

Arti-MANO 的第一个 tracked 机器人手资产位于 `third_party/robot_hands/artimano/`。仓库不分发
Wuji Hand2、外部数据、MANO/SMPL-X 模型或提取缓存。它们应放在仓库外部，
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
| 5 | 完整 GRAB 数据集适配器 | Complete，有界；fresh semantic closeout 通过 | lazy index、原生时间/网格的单序列和双手转换、validation、provenance、raw/binary/官方 semantic contacts 与交互 HOI viewer；全量转换仍不在范围内。 |
| 6 | 物体采样、碰撞几何与 SDF | Complete，有界；假设显式 | mesh audit、确定性 50 点表面参考、仅 collision 的机器人表面采样、SDF 查询、probe、报告、可视化和有界真实数据验收通过；后续交互/优化仍不在范围内。 |
| 7 | 相对骨方向初始化 | Complete，有假设 | 20-bone/15-pair Eq. 1、时序有界 Eq. 2、frame 审计、RH/LH 验收、artifact、验证和可视化通过。 |
| 8 | 交互图与 Laplacian 坐标 | Complete，有界；假设显式 | source-only Eq. 3–7 图/loss、RH/LH artifact、identity/Jacobian 验证、报告和可视化通过；Eq. 8–9 仍属于 Stage 9。 |
| 9 | 带 slack 的受限优化 | Complete，有界；假设显式 | Eq. 8–9 final refinement、full/adaptive QuerySet、slack、独立 full-surface audit、RH/LH artifact、CLI、测试和可视化通过；不包含 Stage 10。 |
| 10 | GRAB→Arti-MANO 端到端重定向 | 已实现；bounded reference-runtime 已接受，preferred 性能仍开放 | 有界可恢复 DAG、官方 contact-window 选择、provenance、review/export，以及已接受的 `s1/airplane_lift` 右手 60 帧 reference-runtime milestone；preferred 性能、production 和 real-time 范围仍开放。 |
| Q1–Q3 | 多数据集交互 benchmark 与统一自动评价 | 已实现，有界；当前本地 ContactPose selection gate 在 freeze 前阻塞 | 冻结 selection contract、metric registry、自动 gate、绑定 manifest 的 profile、报告和 HTML dashboard。当前本地 audit 未发现可识别的 official ContactPose contact attribution，因此未运行 baseline；不声称论文完整 25-grasp ContactPose 结果。 |
| Q4 | morphology-aware warm-start | 未开始 | 在不修改 Q1–Q3 冻结基线的前提下评估 morphology-aware 初始化。 |
| Q5 | Arti-MANO surface contact proxies | 未开始 | 将机器人表面/pad proxy 与 source label 分开验证。 |
| Q6 | contact-aware final extension | 未开始 | 在 Q4/Q5 证据后增加独立版本的 contact-aware 扩展。 |
| Q7 | cross-trajectory 自动 profile 选择 | 未开始 | 只能根据冻结的跨轨迹证据选择 profile。 |
| 12 | OakInk、DexYCB、HO-Cap adapter | TODO | 添加独立验证的数据集 adapter。 |
| 13 | ARCTIC、OakInk2、TACO 扩展 | TODO | 添加独立验证的数据集 adapter。 |
| 14 | 任意灵巧手 plugin interface | TODO | 测试 URDF/MJCF hand plugin contract。 |
| 15 | Baseline 与 ablation | TODO | 添加公平的 OmniRetarget、Mink、DexPilot、GeoRT 运行。 |
| 16 | Reference-tracking PPO | TODO | 添加 RL 训练和评估 pipeline。 |
| 17 | 论文实验复现 | TODO | 复现 tables、figures、seeds 和结果报告。 |
| 18 | 性能优化与 v1.0 release | TODO | 建立 benchmark、打包和 release criteria。 |
| 19 | 非论文扩展 | TODO | 将 MANO 清理、SPIDER 等扩展单独标识。 |

维护中的路线图见 [docs/ROADMAP.md](docs/ROADMAP.md)，中文路线图见
[docs/ROADMAP.zh-CN.md](docs/ROADMAP.zh-CN.md)。benchmark contract、统一自动评价和 Eq. (9)
说明见 [docs/MULTI_DATASET_INTERACTION_BENCHMARK.zh-CN.md](docs/MULTI_DATASET_INTERACTION_BENCHMARK.zh-CN.md)、
[docs/UNIFIED_AUTOMATIC_EVALUATION.zh-CN.md](docs/UNIFIED_AUTOMATIC_EVALUATION.zh-CN.md) 和
[docs/EQ9_TEMPORAL_SCOPE_INTERPRETATIONS.zh-CN.md](docs/EQ9_TEMPORAL_SCOPE_INTERPRETATIONS.zh-CN.md)。

### Q1–Q3 冻结 benchmark

该 benchmark 是有界工程评价，不是论文完整结果复现声明。动态 GRAB clip 与静态 ContactPose
grasp 分开统计；任何 profile 运行前先冻结 selection；ContactPose exact 公式与 GRAB contact
proxy 分表。使用任务环境提供的本机路径或 `.local/config.yaml`：

```bash
export PYTHONNOUSERSITE=1 PYTHONPATH=src
export GRAB_ROOT=/mnt/nas/storage/Ref2Dex_storage/GRAB/data/GRAB
export CONTACTPOSE_ROOT=/mnt/nas/storage/Ref2Dex_storage/ContactPose/data
export MANO_MODEL_ROOT=/mnt/nas/storage/Ref2Dex_storage/shared_assets/body_models/mano
export TOPORETARGET_ARTIMANO_ASSET_ROOT=...    # 可选显式资产 override

python -m toporetarget benchmark inspect-datasets \
  --grab-root "$GRAB_ROOT" --contactpose-root "$CONTACTPOSE_ROOT" \
  --output .local/benchmarks/hoi_benchmark_v1/dataset_audit.json
python -m toporetarget benchmark select --config configs/benchmarks/hoi_benchmark_v1.yaml
python -m toporetarget benchmark freeze
python -m toporetarget benchmark run --resume
python -m toporetarget benchmark evaluate --html
python -m toporetarget benchmark dashboard
```

selection lock 在运行期间不可变，结果不能反过来挑选或替换 unit。当前本地 snapshot 中，固定
clip 加 3 条 additional GRAB selection 通过，但 110 个 ContactPose candidate annotation 都
没有可识别的 official attribution 字段，因此状态为 `Q1_CONTACTPOSE_SELECTION_BLOCKED`，没有
生成 selection manifest、运行 baseline 或声称结果级 metrics；详见 `.local/benchmarks/hoi_benchmark_v1/`。
Q4–Q7 仍未开始。

## 环境配置 Quickstart

### 依赖

- Python 3.10–3.13
- Git
- 使用对应 workflow 时才需要外部数据和模型
- 使用 --show viewer 需要图形 backend；无头 smoke test 可用 MPLBACKEND=Agg

安装当前数据和可视化 workflow 所需的完整环境：

```bash
python -m pip install -e ".[dev,cache,viz,grab,robot,geometry,retarget]"
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

仓库中所有 `--interactive`/`--show` Matplotlib 窗口都使用统一的响应式字体处理：标题、坐标轴
标签、刻度、图例、注释、帧标签和控件标签会随窗口面积放大或缩小；静态 PNG/PDF 输出不变。

### 生成并验证相对骨方向 Warm Start

Stage 7 读取 canonical MediaPipe-21 cache，并写入独立的初始化 artifact。
优化目标不读取 Stage 6 surface samples 或 SDF。

```bash
GRAB_CACHE=.local/cache/hoi/grab/cubemedium_inspect_1_rh_f000000_f000060_mp21.zarr
WARM_START=.local/cache/retarget/warm_start/s7_cubemedium_inspect_1_right_artimano_rh.zarr

toporetarget retarget inspect-bones \
  --canonical "$GRAB_CACHE" --hand right --frame 0 \
  --frame-profile canonical_keypoint_wrist_v1 \
  --bone-profile mediapipe21_full_finger_chain_v1 \
  --json .local/reports/stage7/source_bone_features_right.json \
  --csv .local/reports/stage7/source_bone_features_right.csv

toporetarget retarget compare-frame-profiles \
  --canonical "$GRAB_CACHE" --hand right --robot artimano_rh --frame 0 \
  --report .local/reports/stage7/frame_profile_comparison.json

toporetarget retarget warm-start \
  --canonical "$GRAB_CACHE" --hand right --robot artimano_rh \
  --start-frame 0 --end-frame 60 \
  --frame-profile canonical_keypoint_wrist_v1 \
  --bone-profile mediapipe21_full_finger_chain_v1 \
  --solver-profile paper_repro_scipy_trf --output "$WARM_START"

toporetarget retarget validate-warm-start \
  --canonical "$GRAB_CACHE" --warm-start "$WARM_START" \
  --report .local/reports/stage7/artimano_rh_validation.json \
  --csv .local/reports/stage7/artimano_rh_validation.csv
```

调试时可使用 `visualize-warm-start` 的 `--view scene` 或 `--view local-hand`，
并打开 `--show-directions`、`--show-residuals`、`--show-hand-frames`。详细公式见
[`docs/RELATIVE_BONE_DIRECTION_INITIALIZATION.md`](docs/RELATIVE_BONE_DIRECTION_INITIALIZATION.md)
和 [`docs/WARM_START_OPTIMIZATION.md`](docs/WARM_START_OPTIMIZATION.md)。

交互 viewer 现在复用相同的 scene/local 图层，也支持这些显示参数：

```bash
toporetarget retarget visualize-warm-start \
  --canonical "$GRAB_CACHE" --warm-start "$WARM_START" \
  --view scene --start-frame 0 --end-frame 60 --interactive \
  --show-source-hand --show-robot-hand \
  --show-source-skeleton --show-robot-skeleton \
  --show-hand-frames --show-labels --show-residuals \
  --show-object-context
```

### Stage 7.1：Warm-Start 保真度与可达性审计

manifest 驱动的 Stage 7.1 audit 会重放已接受的 Stage 7 warm-start，检查 source/robot mapping、
thumb URDF ancestry 与 axis、frame/base alignment、joint limits、per-finger attribution 和有界的
diagnostic-only reachability。它把正式 Stage 7 fidelity 与 Stage 8/contact/task fidelity 分开，
不会修改正式 artifact。完整契约和当前 accepted-run 结果见
[`docs/WARM_START_FIDELITY_AND_REACHABILITY_AUDIT.zh-CN.md`](docs/WARM_START_FIDELITY_AND_REACHABILITY_AUDIT.zh-CN.md)。

```bash
env PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/deepcybo/miniconda3/envs/topo-retarget/bin/python \
  -m toporetarget workflow audit-warm-start \
  --run .local/runs/stage10_reference_runtime/s1__airplane_lift__right__artimano_rh__f000240_f000300/manifest.json \
  --canonical-contact-audit .local/runs/stage9_3_2_canonical_reaudit/s1__airplane_lift__right__artimano_rh__f000240_f000300 \
  --output-root .local/runs/stage7_1_warmstart_audit/s1__airplane_lift__right__artimano_rh__f000240_f000300 \
  --html --run-reachability-diagnostics --diagnostic-frames auto
```

### Stage 8：构建并验证共享交互图

Stage 8 将 Stage 7 warm start 和 Stage 6 的 50 点 sample artifact 作为独立、带 hash 检查的输入。
先构建 source-only graph，再用完全相同的 connectivity 和 directed weights 在 robot FK 上冻结评估 Eq. (7)：

```bash
toporetarget retarget audit-interaction-inputs \
  --right-canonical .local/cache/hoi/grab/cubemedium_inspect_1_rh_f000000_f000060_mp21.zarr \
  --left-canonical .local/cache/hoi/grab/s7/cubemedium_inspect_1/semantic_left_f000000_f000060.zarr \
  --right-warm-start .local/cache/retarget/warm_start/s7_cubemedium_inspect_1_right_artimano_rh.zarr \
  --left-warm-start .local/cache/retarget/warm_start/s7_cubemedium_inspect_1_left_artimano_lh.zarr \
  --object-samples .local/cache/geometry/object_surface/cubemedium_samples.npz \
  --report .local/reports/stage8/input_audit.json
```

完整 graph/evaluation 命令与边界见 [`docs/INTERACTION_GRAPH.md`](docs/INTERACTION_GRAPH.md)、
[`docs/LAPLACIAN_INTERACTION_LOSS.md`](docs/LAPLACIAN_INTERACTION_LOSS.md) 和
[`docs/stages/STAGE_8_INTERACTION_GRAPH_LAPLACIAN.md`](docs/stages/STAGE_8_INTERACTION_GRAPH_LAPLACIAN.md)。
Eq. (8)-(9)、slack、SDF/collision penalty 和 optimization 由下一个 Stage 9 workflow 完成。

交互检查 graph、Laplacian、residual 和 contribution（只查看 source graph 时使用 `--mode source`
并省略 `--evaluation`）：

```bash
toporetarget retarget visualize-interaction \
  --graph .local/cache/retarget/interaction_graph/s7_cubemedium_inspect_1_right.zarr \
  --mode compare \
  --evaluation .local/cache/retarget/interaction_evaluation/s7_cubemedium_inspect_1_right_artimano_rh.zarr \
  --start-frame 0 --end-frame 60 --interactive \
  --show-laplacian --show-residuals --show-contributions \
  --report .local/reports/stage8/rh_interactive_viewer.json
```

### Stage 9：生成并验证保交互的最终机器人参考轨迹

Stage 9 使用冻结的 Stage 7 warm start、Stage 8 graph 和 Stage 6 collision-surface artifact，
在显式 local seed-delta 坐标中运行 SLSQP。约束使用 positive-outside signed distance、每个
query 的 slack、单调扩展的 adaptive QuerySet，并用独立的 512 点 full-surface reference audit
验收；solver-only convex-hull backend 只有在与 Stage 6 reference backend probe 一致后才启用。

```bash
toporetarget retarget inspect-query-set \
  --canonical .local/cache/hoi/grab/cubemedium_inspect_1_rh_f000000_f000060_mp21.zarr \
  --warm-start .local/cache/retarget/warm_start/s7_cubemedium_inspect_1_right_artimano_rh.zarr \
  --robot artimano_rh --frame 0 --query-profile adaptive_active_set_v1 \
  --json .local/reports/stage9/rh_query_set_frame0.json

toporetarget retarget refine \
  --canonical .local/cache/hoi/grab/cubemedium_inspect_1_rh_f000000_f000060_mp21.zarr \
  --warm-start .local/cache/retarget/warm_start/s7_cubemedium_inspect_1_right_artimano_rh.zarr \
  --graph .local/cache/retarget/interaction_graph/s7_cubemedium_inspect_1_right.zarr \
  --robot artimano_rh --start-frame 0 --end-frame 60 \
  --output .local/cache/retarget/final/s7_cubemedium_inspect_1_right_artimano_rh.zarr

toporetarget retarget validate-refinement \
  --canonical .local/cache/hoi/grab/cubemedium_inspect_1_rh_f000000_f000060_mp21.zarr \
  --warm-start .local/cache/retarget/warm_start/s7_cubemedium_inspect_1_right_artimano_rh.zarr \
  --final .local/cache/retarget/final/s7_cubemedium_inspect_1_right_artimano_rh.zarr \
  --report .local/reports/stage9/rh_validation.json \
  --csv .local/reports/stage9/rh_validation.csv

toporetarget retarget audit-penetration \
  --canonical .local/cache/hoi/grab/cubemedium_inspect_1_rh_f000000_f000060_mp21.zarr \
  --warm-start .local/cache/retarget/warm_start/s7_cubemedium_inspect_1_right_artimano_rh.zarr \
  --final .local/cache/retarget/final/s7_cubemedium_inspect_1_right_artimano_rh.zarr \
  --robot artimano_rh --report .local/reports/stage9/rh_full_surface_audit.json
```

完整参数、失败策略、RH/LH 命令和验收边界见
[`docs/stages/STAGE_9_FINAL_CONSTRAINED_REFINEMENT.md`](docs/stages/STAGE_9_FINAL_CONSTRAINED_REFINEMENT.md)、
[`docs/FINAL_REFINEMENT_OPTIMIZATION.md`](docs/FINAL_REFINEMENT_OPTIMIZATION.md) 和
[`docs/COLLISION_QUERY_SET_AND_SLACK.md`](docs/COLLISION_QUERY_SET_AND_SLACK.md)。Stage 10、RL、
physics、ContactPose 和 baselines 不在本阶段。

当前 `[0,60)` RH/LH 有界 closeout 已通过 full-surface validation 和 full-artifact determinism。
最小 full signed distance 为 RH `0.623582905 m`、LH `0.641271031 m`，两侧 penetration 均为 0；
`0/29/59` 三帧 adaptive/full 分别使用 16/512 个 query。详细 metrics、hash、Jacobian、solver
comparison 和可视化报告位于被忽略的 `.local/reports/stage9/`；这只关闭 Stage 9，RL、
physics、ContactPose 和 baseline reproduction 仍是 TODO。Stage 10 编排已通过
`toporetarget workflow` 提供；早期 `s7/cubemedium_inspect_1` contact-rich 尝试在既有 Stage 9
solver 的 iteration-limit 处停止，之后的 `s1/airplane_lift` 右手 `[240,300)` run 已作为
bounded reference-runtime milestone 接受，但范围仅限单条离线 60 帧窗口。

#### Stage 9.1 solver-robustness closeout

v1 profile 保持不变。contact-rich run 可能得到可行但 status `9` / `Iteration limit reached`
的 SLSQP candidate；strict acceptance 仍然拒绝它，因为 optimizer convergence 是独立的必要
字段。v2 在 adaptive active-set 扩展时从 `result.x` continuation，按 query ID 映射旧 slack，
只为新 query 初始化有界最小 slack，并保存 continuation trace。它不修改 Eq. (8)、Eq. (9)、
论文权重、base parameterization、q/slack bounds、signed-distance sign 或 full 512-point audit。

Stage 10 resume 时显式选择 profile：

```bash
toporetarget workflow run-grab \
  --sequence s1/airplane_lift --index .local/index/grab \
  --hand right --robot artimano_rh --start-frame 240 --end-frame 300 \
  --refinement-solver-profile scipy_slsqp_active_set_contact_rich_v2 \
  --run-root .local/runs/stage10
```

固定 benchmark、strict status 字段、deterministic repeat、最终统一 maxiter 及 profile hash
记录在 `.local/reports/stage9_1/maxiter_benchmark.json`。Stage 10 signature 包含所选 profile
ID/hash；切换 profile 只使 Stage 9 及下游节点失效，Stage 5–8 input 可复用。solver/termination
细节仍属于论文未公开的实现假设。
当前固定 benchmark 共 35 条记录，选择统一 `maxiter=100`。v1 hash 为
`6affff2fdb425a0402f643c291c0b8904d4dbec6c5b69a5006cf9829dcc220aa`，v2 hash 为
`c42c21d894c54d07b1d30943b5a3338b13628bf0429ab203b5540cf934d09b7c`。完整 60 帧真实
artifact 和 deterministic repeat 已由 Stage 9.2 contact-rich run 及 full fresh/resumed comparison
支撑。Stage 9.2 通过 reference-runtime minimum gate，但 preferred single-frame median/p95
目标仍未达到；bounded reference-runtime Stage 10 milestone 已接受，但 preferred 性能债务仍开放。

#### Stage 9.2 性能与可恢复执行

Stage 9.2 增加 exact-x callback reuse、持久 SDF/FK resource、精确 reference-SDF AABB
加速、批量 collision Jacobian、显式 solver conditioning、独立 full-512 audit 调度、原子 frame checkpoint、soft wall-time pause/resume、assembly
和 fresh/resumed comparison，同时保持 Stage 9 数学及 strict status-9 policy 不变。
Profiling 和恢复命令见 [`docs/REFINEMENT_PERFORMANCE.md`](docs/REFINEMENT_PERFORMANCE.md)
及 [`docs/REFINEMENT_CHECKPOINT_AND_RESUME.md`](docs/REFINEMENT_CHECKPOINT_AND_RESUME.md)。
完整 60 帧 minimum runtime gate 与 deterministic repeat 证据已通过。v3 第一次/重复运行
的 median/p95 分别为 `10.766/38.711 s` 与 `10.773/39.052 s`，两次均为 `60/60`
strict-accepted status-0 frame；排除 `solve_time_s` 后持久化数组 exact equal。bounded
reference-runtime Stage 10 milestone 已接受；preferred 性能 gate 以及 production/real-time
范围仍未完成。

### Stage 10：运行有界 GRAB → Arti-MANO workflow

当前已物化 bounded reference-runtime milestone：
`.local/runs/stage10_reference_runtime/s1__airplane_lift__right__artimano_rh__f000240_f000300/`。
该 run 只复用已接受的 Stage 5–9 artifact，`solver_invocation_count=0`，并导出
`exports/robot_reference.zarr` 与 NPZ。该决定仅适用于单条离线 60 帧 milestone，
不代表全数据集、production、real-time 或 online control readiness。

```bash
toporetarget workflow run-grab \
  --sequence s1/airplane_lift --index .local/index/grab \
  --hand right --robot artimano_rh --auto-contact-window --window-length 60 \
  --mano-model-root /path/to/MANO --asset-root third_party/robot_hands/artimano \
  --run-root .local/runs/stage10 \
  --manual-acceptance .local/reports/stage9/manual_acceptance.json
```

生成 manifest 后可使用 `workflow status`、`workflow validate`、`workflow visualize` 和
`workflow export-reference`；断点续跑及 provenance 规则见
[`docs/WORKFLOW_RESUME_AND_PROVENANCE.md`](docs/WORKFLOW_RESUME_AND_PROVENANCE.md)。

如需在浏览器中查看真实的 source MANO mesh，以及 warm-start 和 final 的 Arti-MANO visual
mesh，可生成自包含 HTML：

```bash
toporetarget workflow visualize-mesh \
  --run .local/runs/stage10_reference_runtime/s1__airplane_lift__right__artimano_rh__f000240_f000300/manifest.json \
  --interactive
```

默认输出为 `review/trajectory_mesh.html`。蓝色、橙色、绿色分别表示 source、warm-start、final
mesh；页面支持逐帧播放、旋转/缩放、object 上下文点云和逐帧 refinement 指标。它只是可视化检查工具，
不能替代 Stage 9/10 的数值 gate。

同一个页面还包含冻结的 Stage 8 interaction graph 和 Laplacian 诊断。可以在页面下拉框中切换，
也可以在生成时指定初始模式：

```bash
toporetarget workflow visualize-mesh \
  --run .local/runs/stage10_reference_runtime/s1__airplane_lift__right__artimano_rh__f000240_f000300/manifest.json \
  --mode combined \
  --output .local/runs/stage10_reference_runtime/s1__airplane_lift__right__artimano_rh__f000240_f000300/review/trajectory_combined.html
```

可用模式为 `mesh`、`full-graph`、`figure4-style`、`laplacian-diagnostic` 和 `combined`；所有模式都会保留 source/warm-start/final mesh。
`figure4-style` 默认只显示 hand-object 边；侧栏还可控制 edge threshold/top-k、权重显示方式、
residual target/scope、标量热度、向量箭头、labels 以及 source/warm/final graph 状态。图结构和
object samples 均直接读取已接受的 Stage 8 artifact；该 viewer 不重建也不修改它们。
HTML 中额外点/线、residual 叠加层、右侧面板各选项及查看方法的完整说明见
[`docs/INTERACTION_MESH_VISUALIZATION.md`](docs/INTERACTION_MESH_VISUALIZATION.md)。

Stage 9 有界 clip 的交互查看（不使用 `--output`）：

```bash
toporetarget retarget visualize-refinement \
  --canonical .local/cache/hoi/grab/cubemedium_inspect_1_rh_f000000_f000060_mp21.zarr \
  --warm-start .local/cache/retarget/warm_start/s7_cubemedium_inspect_1_right_artimano_rh.zarr \
  --graph .local/cache/retarget/interaction_graph/s7_cubemedium_inspect_1_right.zarr \
  --final .local/cache/retarget/final/s7_cubemedium_inspect_1_right_artimano_rh.zarr \
  --robot artimano_rh --start-frame 0 --end-frame 60 --interactive \
  --show-labels --show-frames --show-collision-samples --show-query-set \
  --show-penetrations --show-slack --report .local/reports/stage9/rh_interactive_viewer.json
```

#### 查看整条轨迹

现有 `f000000_f000060` 输入和 final artifact 只有半开区间 `[0,60)`；`visualize-refinement` 无法
恢复没有转换、没有优化的帧。要查看完整源序列，先生成一个 full canonical artifact，再在 Stage 7–9
中省略帧范围参数。下面是 RH 的完整流程；LH 将 `right`/`artimano_rh` 替换为
`left`/`artimano_lh`，并使用对应的 collision-surface artifact。流程只读复用现有 Stage 6
object 和 collision-surface artifact。

```bash
export FULL_CANONICAL=.local/cache/hoi/grab/s7/cubemedium_inspect_1/both_full_mp21.zarr
export OBJECT_SAMPLES=.local/cache/geometry/object_surface/cubemedium_samples.npz
export RH_WARM_FULL=.local/cache/retarget/warm_start/s7_cubemedium_inspect_1_right_artimano_rh_full.zarr
export RH_GRAPH_FULL=.local/cache/retarget/interaction_graph/s7_cubemedium_inspect_1_right_full.zarr
export RH_EVAL_FULL=.local/cache/retarget/interaction_evaluation/s7_cubemedium_inspect_1_right_full.zarr
export RH_FINAL_FULL=.local/cache/retarget/final/s7_cubemedium_inspect_1_right_artimano_rh_full.zarr
export RH_SURFACE=.local/cache/geometry/robot_surface/artimano_rh_neutral.npz

# 1. 转换并验证所有帧。省略两个 frame 参数即表示完整序列。
toporetarget data convert --dataset grab --index .local/index/grab \
  --sequence s7/cubemedium_inspect_1 --hands both \
  --include-table --contact-mode semantic --include-mediapipe21 \
  --mano-model-root "$MANO_MODEL_ROOT" --output "$FULL_CANONICAL" --force
toporetarget data validate --dataset grab --index .local/index/grab \
  --sequence s7/cubemedium_inspect_1 --canonical "$FULL_CANONICAL" \
  --mano-model-root "$MANO_MODEL_ROOT" \
  --report .local/reports/stage5/grab_full_validation.json
toporetarget geometry validate-samples --canonical "$FULL_CANONICAL" \
  --object-id primary --samples "$OBJECT_SAMPLES" \
  --report .local/reports/stage6/full_object_samples_validation.json

# 2. Stage 7 完整 warm start。
toporetarget retarget warm-start --canonical "$FULL_CANONICAL" \
  --hand right --robot artimano_rh \
  --frame-profile canonical_keypoint_wrist_v1 \
  --bone-profile mediapipe21_full_finger_chain_v1 \
  --solver-profile paper_repro_scipy_trf --output "$RH_WARM_FULL" --force

# 3. Stage 8 完整 graph 和冻结的 Eq. (7) evaluation。
toporetarget retarget build-interaction-graph --canonical "$FULL_CANONICAL" \
  --hand right --object-samples "$OBJECT_SAMPLES" \
  --output "$RH_GRAPH_FULL" --report .local/reports/stage8/rh_full_graph_build.json --force
toporetarget retarget evaluate-interaction --graph "$RH_GRAPH_FULL" \
  --warm-start "$RH_WARM_FULL" --robot artimano_rh \
  --output "$RH_EVAL_FULL" --force

# 4. Stage 9 顺序执行完整 refinement、validation 和独立碰撞审计。
toporetarget retarget refine --canonical "$FULL_CANONICAL" \
  --warm-start "$RH_WARM_FULL" --graph "$RH_GRAPH_FULL" --robot artimano_rh \
  --query-profile adaptive_active_set_v1 --coordinate-profile local_seed_delta_v1 \
  --solver-profile scipy_slsqp_active_set_v1 --collision-samples "$RH_SURFACE" \
  --output "$RH_FINAL_FULL" --force
toporetarget retarget validate-refinement --canonical "$FULL_CANONICAL" \
  --warm-start "$RH_WARM_FULL" --graph "$RH_GRAPH_FULL" --final "$RH_FINAL_FULL" \
  --robot artimano_rh --collision-samples "$RH_SURFACE" \
  --report .local/reports/stage9/rh_full_validation.json
toporetarget retarget audit-penetration --canonical "$FULL_CANONICAL" \
  --warm-start "$RH_WARM_FULL" --final "$RH_FINAL_FULL" --robot artimano_rh \
  --collision-samples "$RH_SURFACE" \
  --report .local/reports/stage9/rh_full_surface_audit.json

# 5. 省略范围时，viewer 默认读取 final artifact 的全部帧。
toporetarget retarget visualize-refinement --canonical "$FULL_CANONICAL" \
  --warm-start "$RH_WARM_FULL" --graph "$RH_GRAPH_FULL" --final "$RH_FINAL_FULL" \
  --robot artimano_rh --interactive --show-labels --show-frames \
  --show-collision-samples --show-query-set --show-penetrations --show-slack
```

索引中的 `s7/cubemedium_inspect_1` 一共有 951 帧、120 FPS；当前有界 clip 只有 0–59 帧。当前工作站上
有界的 60 帧 RH/LH 运行每侧约 20 分钟，因此完整 Stage 9 粗略需要每侧 5 小时 17 分钟、两侧约
10 小时 34 分钟，不含转换和上游阶段。这只是线性估计，应以逐帧诊断为准。viewer 不会再次运行
solver，只读取 final artifact。如果完整流程后手和物体仍然
相距很远，应检查 canonical scene overlay，以及 final viewer 中的 object、collision-sample、query-set、
penetration 和 slack 图层。不要靠扩大 viewer 范围或改 object pose/sample 人为制造碰撞；较大的正 SDF
和零 penetration 表示该源轨迹在该帧确实没有碰撞。

如需生成 first/middle/last 静态诊断图，只需修改 `--frame` 和 `--output`：

```bash
# Scene overlay：source/robot keypoints、skeleton、frame、residual 和 object context。
toporetarget retarget visualize-warm-start \
  --canonical "$GRAB_CACHE" --warm-start "$WARM_START" \
  --view scene --frame 0 \
  --show-hand-frames --show-labels --show-residuals --show-object-context \
  --output .local/reports/stage7/scene_first.png

# Local wrist-centered overlay：骨方向和 adjacent features。
toporetarget retarget visualize-warm-start \
  --canonical "$GRAB_CACHE" --warm-start "$WARM_START" \
  --view local-hand --frame 0 \
  --show-directions --show-adjacent-features --show-labels --show-residuals \
  --output .local/reports/stage7/local_first.png
```

将 `--frame` 改为 `30` 和 `59` 可生成 middle/last 图。交互窗口中的 keypoint、skeleton、
frame、label、residual 字体会随窗口缩放；`--show-object-context` 只用于显示，不会进入
warm-start 优化目标。

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

交互式 comparison 窗口：

```bash
toporetarget data compare \
  --dataset synthetic --sequence demo \
  --canonical .local/cache/hoi/synthetic_demo.zarr \
  --layout side-by-side --start-frame 0 --end-frame 8 --show \
  --show-keypoints --show-mesh --show-scene-frame --show-object-frame
```

帧范围是连续的半开区间：--start-frame 0 --end-frame 60 表示 0–59 帧。
--show 是交互模式，--output 生成无头 PNG。

### 2. GRAB NPZ → canonical Zarr

历史 Stage 2B GRAB reader 按单条明确序列工作：读取一个 NPZ 并选择一只手；生产级 Stage 5
adapter 见下文。生成完整序列时不要指定 --start-frame 和 --end-frame；
生成片段时同时指定二者。

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

### 5. 生产级 GRAB 数据集 adapter

构建 filename-first index，在不加载帧数组的情况下查询，并在保留 source timestamps、原生
mesh、个性化 MANO `vtemp`、object/table pose 和 source/binary/官方 semantic contacts 的前提下转换单条右手、
左手或双手序列：

```bash
toporetarget data index --dataset grab --output .local/index/grab
toporetarget data list --dataset grab --index .local/index/grab --subject s7 --limit 20
toporetarget data describe --dataset grab --index .local/index/grab \
  --sequence s7/cubemedium_inspect_1
toporetarget data convert --dataset grab --index .local/index/grab \
  --sequence s7/cubemedium_inspect_1 --hands both --start-frame 0 --end-frame 60 \
  --include-table --contact-mode source --include-mediapipe21 \
  --mano-model-root "$MANO_MODEL_ROOT" \
  --output .local/cache/hoi/grab/s7/cubemedium_inspect_1/both_f000000_f000060.zarr
toporetarget data validate --dataset grab --index .local/index/grab \
  --sequence s7/cubemedium_inspect_1 \
  --canonical .local/cache/hoi/grab/s7/cubemedium_inspect_1/both_f000000_f000060.zarr \
  --mano-model-root "$MANO_MODEL_ROOT" \
  --report .local/reports/stage5/grab_validation.json
```

使用 `--contact-mode semantic` 会保留原始 GRAB label、派生 binary mask，并附加
`configs/datasets/grab_contact_parts.yaml` 中验证过的官方 0--55 body/hand mapping：

```bash
toporetarget data convert --dataset grab --index .local/index/grab \
  --sequence s7/cubemedium_inspect_1 --hands both --start-frame 0 --end-frame 60 \
  --include-table --contact-mode semantic --include-mediapipe21 \
  --mano-model-root "$MANO_MODEL_ROOT" \
  --output .local/cache/hoi/grab/s7/cubemedium_inspect_1/semantic_f000000_f000060.zarr
```

adapter 不做时间重采样、空间/物体表面采样、raw source 写入或全量转换。使用
`toporetarget data visualize` 可进行 raw/canonical/compare、overlay 或 side-by-side、帧
slider/键盘播放、scene/object/wrist reference、semantic contact 颜色和无头 PNG 输出。
标准参数名是 `--reference-frame`；旧的 `--reference` 仍可用但会给出 deprecated warning。详见
[docs/GRAB_DATASET_ADAPTER.md](docs/GRAB_DATASET_ADAPTER.md) 与
[docs/GRAB_INTERACTIVE_VISUALIZATION.md](docs/GRAB_INTERACTIVE_VISUALIZATION.md)。

对有界 clip 打开 canonical scene 的交互检查：

```bash
toporetarget data visualize --dataset grab --index .local/index/grab \
  --sequence s7/cubemedium_inspect_1 \
  --canonical .local/cache/hoi/grab/s7/cubemedium_inspect_1/both_f000000_f000060.zarr \
  --mode canonical --reference-frame scene --start-frame 0 --end-frame 60 \
  --interactive --show-mediapipe21 --show-mesh --show-table --show-contacts --show-axes
```

### 6. Tracked Arti-MANO 资产设置

默认运行使用仓库内的 tracked snapshot。先查看解析来源和 provenance：

```bash
toporetarget robots resolve-assets
toporetarget robots compare-assets \
  --reference-root .local/assets/artimano
```

从固定版本的 ManipTrans checkout 重建 tracked snapshot：

```bash
toporetarget assets vendor-artimano \
  --source-root "$MANIPTRANS_ROOT" \
  --destination third_party/robot_hands/artimano \
  --imported-at 2026-07-27T19:00:00+08:00
```

旧的 `import-artimano` 命令与 `.local/assets/artimano` 只用于兼容和迁移；fallback 时会输出 deprecation
warning。见 [docs/TRACKED_ROBOT_HAND_ASSETS.md](docs/TRACKED_ROBOT_HAND_ASSETS.md)、
[docs/THIRD_PARTY_ASSET_POLICY.md](docs/THIRD_PARTY_ASSET_POLICY.md)、
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

### 7. 目标机器人手资产配置与运动学检查

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
inspect 和 validate 从 `--asset-root`、`TOPORETARGET_ARTIMANO_ASSET_ROOT`、tracked snapshot 或
legacy fallback 解析资产根目录；可用 `toporetarget robots resolve-assets` 查看最终来源。

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

要在本地打开同一 neutral collision 窗口，将 `--output ...png` 替换为 `--show`；窗口文字同样会
随窗口响应式缩放。

接口会报告缺失的 collision geometry，不会静默生成替代几何。Stage 4 将 `palm` 定义为工程
URDF base frame，不选择论文中尚未确定的 wrist frame 参数化，也不执行 MANO→Arti-MANO
重定向。详见 [docs/ROBOT_HAND_INTERFACE.md](docs/ROBOT_HAND_INTERFACE.md) 和
[docs/ARTIMANO_ADAPTER.md](docs/ARTIMANO_ADAPTER.md)。

### 8. 检查对象几何、生成表面参考点并验证 Signed Distance

该有界 geometry workflow 复用现有 canonical object-local mesh 和 Stage 4 collision geometry
contract。论文固定对象点数为 50；采样器、seed、temporal reuse、normals、SDF backend 和机器
人 collision 点数均作为显式 engineering assumption 记录。

```bash
toporetarget geometry inspect-mesh \
  --canonical .local/cache/hoi/grab/s7/cubemedium_inspect_1/right_f000000_f000060.zarr \
  --object-id primary --json .local/reports/stage6/grab_object_mesh_audit.json
toporetarget geometry sample-object \
  --canonical .local/cache/hoi/grab/s7/cubemedium_inspect_1/right_f000000_f000060.zarr \
  --object-id primary --profile paper_strict_area_uniform \
  --output .local/cache/geometry/object_surface/object.npz \
  --report .local/reports/stage6/object_samples.json
toporetarget geometry validate-samples --canonical .local/cache/hoi/grab/s7/cubemedium_inspect_1/right_f000000_f000060.zarr \
  --object-id primary --samples .local/cache/geometry/object_surface/object.npz \
  --report .local/reports/stage6/object_samples_validation.json
toporetarget geometry validate-sdf --shape sphere \
  --report .local/reports/stage6/sdf_sphere_validation.json
toporetarget geometry sample-robot --robot artimano_rh --pose neutral \
  --profile engineering_collision_32_per_geometry \
  --output .local/cache/geometry/robot_surface/artimano_rh.npz
toporetarget geometry probe-collision \
  --robot-samples .local/cache/geometry/robot_surface/artimano_rh.npz \
  --object-shape cube --report .local/reports/stage6/synthetic_collision_probe.json

# 可视化固定的 50 个对象采样点及其 ID/法向。
toporetarget geometry visualize-object \
  --canonical .local/cache/hoi/grab/s7/cubemedium_inspect_1/right_f000000_f000060.zarr \
  --object-id cubemedium \
  --samples .local/cache/geometry/object_surface/cubemedium_samples.npz \
  --frame 0 \
  --output .local/reports/stage6/object_samples_frame0_ids.png \
  --show-ids --show-normals --show-object-frame --show-scene-frame

# 将 --frame 改为 29 和 59，可生成 middle/last frame overlay。

# SDF slice 和机器人 collision surface 的静态诊断图。
toporetarget geometry visualize-sdf \
  --shape sphere --slice-axis z --slice-value 0 \
  --output .local/reports/stage6/sdf_sphere_slice_z0.png
toporetarget geometry visualize-robot-surface \
  --robot artimano_rh --pose neutral \
  --profile engineering_collision_32_per_geometry \
  --samples .local/cache/geometry/robot_surface/artimano_rh_neutral.npz \
  --output .local/reports/stage6/artimano_rh_collision_surface.png \
  --show-sample-normals
```

对象 viewer 会显示固定的 50 个 sample ID 和法向；使用 `--frame 29`、`--frame 59` 可生成
middle/last frame overlay，这些帧复用相同的 face+barycentric identity，只改变 object pose。
其他 debug visualization 包括 SDF slice 和 RH/LH collision surface。详见
[OBJECT_GEOMETRY_AND_SAMPLING.md](docs/OBJECT_GEOMETRY_AND_SAMPLING.md)、
[SIGNED_DISTANCE_AND_COLLISION_QUERIES.md](docs/SIGNED_DISTANCE_AND_COLLISION_QUERIES.md) 和
[Stage 6 报告](docs/stages/STAGE_6_OBJECT_GEOMETRY_SDF.md)。

### 9. 论文追踪与复现审计

```bash
python scripts/check_paper_fidelity.py
toporetarget doctor paper
```

论文 PDF、公式/表格/图追踪和未解决假设见
[docs/PAPER_FIDELITY.md](docs/PAPER_FIDELITY.md)、
[docs/PAPER_FIDELITY.yaml](docs/PAPER_FIDELITY.yaml) 和
[docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md)。

### 10. 开发验证

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

## Stage 9.3 接触保持审计

该审计是基于 manifest、不会调用 solver 的诊断流程，比较 accepted Stage 9.2/
Stage 10 artifact 中的 source、warm-start、final、visual robot geometry、collision
geometry、QuerySet provenance、同定义 objective，以及不执行优化的 warm-to-final
插值路径。

```bash
conda run -n topo-retarget env PYTHONNOUSERSITE=1 \
  python -m toporetarget workflow audit-contact-retention \
  --run .local/runs/stage10_reference_runtime/<run>/manifest.json \
  --output-dir .local/runs/stage9_3_contact_audit/<run> \
  --surface-samples 8192 --thresholds-mm 1,2,3,5,8,10 --html --force
```

输出记录输入 artifact 的 hash/mtime 不变性、positive-outside signed-distance 约定、
proxy 假设、per-frame/per-link CSV、root-cause 分析和 self-contained HTML。Source
contact 与 semantic-anchor retention 明确只是诊断 proxy，不是真实接触标签。详见
[`docs/CONTACT_RETENTION_AUDIT.md`](docs/CONTACT_RETENTION_AUDIT.md) 和本页中文说明。

关于 signed-distance reconciliation 与最多三个 frame 的 fail-closed shadow boundary，见
[`docs/CONTACT_METRIC_RECONCILIATION_AND_SHADOW_ABLATION.md`](docs/CONTACT_METRIC_RECONCILIATION_AND_SHADOW_ABLATION.md)。
当前 reference-runtime closeout 因旧 Stage 9.3 convex-hull metric 与 Stage 9.2 reference
SDF 定义不一致而阻止 shadow 执行。

### Stage 9.3.2 canonical re-audit

正式 audit 来源是版本化的
[`reference_winding_v1`](configs/audit/contact_distance/reference_winding_v1.yaml)
reference winding SDF。solver-side SDF acceleration 与 formal evaluation 分离；旧的
convex-hull 报告只用于诊断，并对正式 contact/penetration claim 标记为 superseded。
v2 audit 将 raw penetration、tau/hard/soft residual、visual approximation、collision
distance 和 contact-retention proxy 分开记录。开放 visual mesh 可以支持 unsigned coverage
审计，但不能证明 inflated/inset 方向。Shadow profile 只是 diagnostic，且必须在 canonical
60x512 gate 通过后才可运行；本阶段不实现 Stage 9.4，也不修改 Stage 10 artifact。详见
[`docs/CANONICAL_CONTACT_DISTANCE_AND_REAUDIT.md`](docs/CANONICAL_CONTACT_DISTANCE_AND_REAUDIT.md)。

## 文档索引

- [Roadmap](docs/ROADMAP.md) / [中文路线图](docs/ROADMAP.zh-CN.md)
- [统一 HOI 接口](docs/HOI_DATA_INTERFACE.md)
- [坐标约定](docs/COORDINATE_CONVENTIONS.md)
- [GRAB 检查](docs/GRAB_INSPECTION.md)
- [GRAB 数据集 adapter](docs/GRAB_DATASET_ADAPTER.md) / [交互可视化](docs/GRAB_INTERACTIVE_VISUALIZATION.md)
- [MANO→MediaPipe21 adapter](docs/MANO_TO_MEDIAPIPE21.md)
- [通用机器人手接口](docs/ROBOT_HAND_INTERFACE.md)
- [Arti-MANO 目标手适配器](docs/ARTIMANO_ADAPTER.md)
- [Stage 4 报告](docs/stages/STAGE_4_ARTIMANO_TARGET_HAND.md)
- [Stage 5 报告](docs/stages/STAGE_5_GRAB_DATASET_ADAPTER.md)
- [对象几何与采样](docs/OBJECT_GEOMETRY_AND_SAMPLING.md)
- [Signed Distance 与碰撞查询](docs/SIGNED_DISTANCE_AND_COLLISION_QUERIES.md)
- [Stage 6 报告](docs/stages/STAGE_6_OBJECT_GEOMETRY_SDF.md)
- [Interaction graph](docs/INTERACTION_GRAPH.md) / [中文交互图](docs/INTERACTION_GRAPH.zh-CN.md)
- [Laplacian interaction loss](docs/LAPLACIAN_INTERACTION_LOSS.md) /
  [中文 Laplacian loss](docs/LAPLACIAN_INTERACTION_LOSS.zh-CN.md)
- [Stage 8 报告](docs/stages/STAGE_8_INTERACTION_GRAPH_LAPLACIAN.md) /
  [中文 Stage 8 报告](docs/stages/STAGE_8_INTERACTION_GRAPH_LAPLACIAN.zh-CN.md)
- [论文忠实度](docs/PAPER_FIDELITY.md)
- [数据与许可证策略](docs/LICENSE_AND_DATA_POLICY.md)
- [开发日志](docs/DEVELOPMENT_LOG.md) / [中文开发日志](docs/DEVELOPMENT_LOG.zh-CN.md)
- [贡献指南](CONTRIBUTING.md) / [第三方声明](THIRD_PARTY_NOTICES.md)

## License

仓库代码和文档使用 GNU General Public License v3.0，见 [LICENSE](LICENSE)。tracked Arti-MANO 的
许可证和 notice 保存在 `third_party/robot_hands/artimano/`；外部 GRAB、MANO/SMPL-X、ManipTrans
源码和其他数据集继续遵循其自身许可证。
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

## Stage 9.3.3 shadow 等价性

诊断性的 Stage 9.3.3 boundary 见
[`docs/SHADOW_EQUIVALENCE_AND_LONG_FINGER_ABLATION.zh-CN.md`](docs/SHADOW_EQUIVALENCE_AND_LONG_FINGER_ABLATION.zh-CN.md)。
必须先通过 official numerical-equivalence gate，才能运行六个 bounded shadow
profile；同时保留 Eq. (1)-(9)、Stage 9.2 和 Stage 10 boundary。当前 accepted-window
replay 为 fail-closed：`SHADOW_BASELINE_NOT_NUMERICALLY_EQUIVALENT`；mandatory
shadow profile 和 Stage 9.4 implementation 均未运行/授权。

## Stage 9.3.4 provenance-rebased 因果实验

Stage 9.3.4 是只读审计层，单独分离 historical lane 与 current-lineage
baseline，并运行有界 multistart/base-seed 诊断及保守的 Stage 9.4 路由。详见
[`docs/STAGE9_PROVENANCE_MULTISTART_AND_CAUSAL_ABLATION.md`](docs/STAGE9_PROVENANCE_MULTISTART_AND_CAUSAL_ABLATION.md)。

## Stage 9.3.5 Projection 与因果闭环

Stage 9.3.5 增加 warm→final 可行性扫描、诊断 projection、状态
counterfactual、目标/约束归因和 gated branch rollout。它仅用于诊断，保留
Eq. (1)-(9)、正式权重、正式 artifact 与 Stage 10。实际命令和输出目录见同步文档
[`docs/PROJECTION_FEASIBILITY_AND_CAUSAL_CLOSURE.zh-CN.md`](docs/PROJECTION_FEASIBILITY_AND_CAUSAL_CLOSURE.zh-CN.md)。

## Stage 9 一次性因果闭环与修复

固定 C0--C7 因果 sweep、Eq. (9) 审计、单一 faithful 修复、完整 60 帧验证以及
版本化 Stage 10 review bundle 见
[`docs/STAGE9_ONE_SHOT_CAUSAL_CLOSURE_AND_REPAIR.md`](docs/STAGE9_ONE_SHOT_CAUSAL_CLOSURE_AND_REPAIR.md)。

## Faithful reproduction 正式收口

已接受的 canonical faithful v3-fixed profile、legacy v2 分类、质量中性人工验收、
已正式收口的 versioned fixed Stage 10 export 以及 A/B/C 决策语义见
[`docs/FAITHFUL_REPRODUCTION_FINALIZATION.zh-CN.md`](docs/FAITHFUL_REPRODUCTION_FINALIZATION.zh-CN.md)。

## GRAB Arti-MANO 质量 A–E 实验

固定的四条质量实验轨迹已由 `toporetarget quality` 实现：四条轨迹都来自
`s1`，保持原生帧率，并保留两个 paper-core Eq. (9) profile。所有新 artifact
只写入 `.local/experiments/grab_artimano_quality_v1/`。

```bash
PYTHONNOUSERSITE=1 /home/deepcybo/miniconda3/envs/topo-retarget/bin/python \
  -m toporetarget quality run-a-to-e \
  --config configs/experiments/grab_artimano_quality_v1.yaml \
  --resume --max-wall-time 1800 --generate-html
```

使用 `toporetarget quality status` 查看自动推荐，并打开 `html/index.html`
查看四个自包含 viewer。GRAB contact 只属于 dataset proxy，ContactPose
明确 deferred；结论仅限于 within-subject multi-object development benchmark。
完整边界见
[`docs/GRAB_ARTIMANO_QUALITY_EXPERIMENT.md`](docs/GRAB_ARTIMANO_QUALITY_EXPERIMENT.md)。

对于 open-object geometry，质量 lane 使用文档化的
[`hybrid_original_distance_proxy_sign_v1`](docs/HYBRID_SIGNED_DISTANCE_FOR_OPEN_OBJECTS.md)：
原始网格保持不可变，original mesh 提供 distance magnitude 和 closest point，
derived watertight proxy 只提供 sign。当前 banana run 在严格 active-QuerySet
boundary gate 处正式路由为 `SIGN_PROXY_CONTACT_REGION_CONFLICT`，不是 A–E 完成
声明。详见
[`docs/DERIVED_WATERTIGHT_SIGN_PROXY.md`](docs/DERIVED_WATERTIGHT_SIGN_PROXY.md)。
