# TopoRetarget-Repro

[English README](README.md)

TopoRetarget-Repro 是论文
[*TopoRetarget: Interaction-Preserving Retargeting for Dexterous Manipulation*](https://arxiv.org/abs/2606.16272)
的非官方、独立且可追踪论文依据的复现仓库。它通过 canonical HOI contract、MANO 语义转换、
目标机器人手运动学、geometry/SDF、相对骨方向 warm start、交互图、受限 refinement、验证和
manifest 绑定的导出，将 GRAB 手物交互运动转换为离线灵巧手 reference。

仓库支持 tracked Arti-MANO 与 Wuji Hand2 Beta1 目标手资产，不分发外部 dataset 或
MANO/SMPL-X model。当前实现是带显式 provenance 与假设的有界工程复现，不声称
author-exact、全数据集、实时、真实硬件控制、physics 或 RL 复现。

## 仓库概览

`toporetarget` CLI 提供：

- 保留 native timestamp、scene-frame geometry 与显式 SE(3) 约定的机器人无关
  `HOISequence` schema；
- lazy GRAB index、有界 native-frame 转换、semantic contact 处理及
  MANO→MediaPipe-style-21 转换；
- YAML 注册的通用 URDF 目标手、differentiable/reference FK、named qpos、semantic anchor、
  collision surface 以及 tracked Arti-MANO/Wuji 资产；
- 确定性 object-surface 采样、signed-distance query、collision QuerySet 和独立 full-surface
  audit；
- relative-bone warm start、冻结的 source interaction graph/Laplacian 以及受限的
  interaction-preserving refinement；
- 可恢复、content-hashed workflow、不可变 source/provenance 记录、自动验证、人工 review
  边界和版本化 robot-reference export；
- 覆盖 source/warm/final mesh、interaction graph、continuity、collision/contact 诊断、
  metric 和审计证据的 self-contained browser HTML。

机器本地 dataset、model、cache、report 与 run 应位于被忽略的 `.local/`。核心契约见
[HOI 数据接口](docs/HOI_DATA_INTERFACE.md)、
[坐标约定](docs/COORDINATE_CONVENTIONS.md)和
[机器人手目标契约](docs/ROBOT_HAND_TARGET_CONTRACT.md)。

## Isaac Lab GPU Backend

MuJoCo 保留为 CPU correctness、deterministic regression、contact diagnostic、
action replay 与 visualization 后端。GPU 并行平台与后续 policy 工作迁移到独立的
Isaac Lab 路径；MuJoCo 证据不能授权 PhysX 资产、Oracle 或 PPO。

Stage 16-C.0 冻结 Python 3.11.15、Isaac Sim 5.1.0、Isaac Lab `v2.3.2`
（commit `37ddf626871758333d6ed89cf64ad702aef127d0`）与 Torch 2.7.0 cu128。
在用户明确给出进程级 EULA 授权后，真实 GPU 资格结果为
`STAGE16C0_ISAACLAB_PLATFORM_VALIDATED_WITH_LIMITATIONS`：全部硬 gate 通过；
缺少交互显示与上游依赖 metadata 冲突作为软限制保留。

NVIDIA 当前已将 Isaac Sim 5.1 标记为 unsupported；这里的 5.1/v2.3.2 组合是冻结的
复现目标，不代表厂商仍提供持续支持。

```bash
bash scripts/bootstrap_stage16_isaaclab_env.sh --dry-run
conda run -n toporetarget-isaaclab python scripts/verify_stage16_isaaclab_platform.py --phase static
conda run -n toporetarget-isaaclab python scripts/verify_stage16_isaaclab_platform.py --phase full --steps 1000 --accept-eula
conda run -n toporetarget-isaaclab python scripts/rl/isaaclab/validate_stage16c1_assets.py
conda run -n toporetarget-isaaclab python scripts/rl/isaaclab/import_wuji_hand2.py --upstream-root /home/deepcybo/workspace/dex/wuji-description --accept-eula
conda run -n toporetarget-isaaclab python scripts/rl/isaaclab/import_hocap_objects.py --accept-eula
```

`--accept-eula` 仅在用户明确授权后可用，并且只在启动的运行时进程中设置
`OMNI_KIT_ACCEPT_EULA=YES`；它不代表隐私或 telemetry 同意。Stage 16-C.1 已验证
浮动基座 Wuji articulation、两个 free HO-Cap rigid object、1/128-env CUDA smoke
与有界接触响应。生成的 USD 和报告仍位于被忽略的 `.local/`。自定义
`DirectRLEnv`、PhysX Oracle 与 PPO 尚未开始。详见
[资产迁移契约](docs/rl/ISAACLAB_ASSET_MIGRATION.md)。

## 数据集支持

| 数据集 | Adapter | Source qualification | Strict final qualification | 说明 |
|---|---|---:|---:|---|
| GRAB | 完成 | 已验证 | 已验证 | 初始动态 reference dataset |
| DexYCB | 完成 | 2/2 | 2/2 | 原生 PCA45 与 subject-shape 路由 |
| OakInk | 完成 | 2/2 | 2/2 | 原生 hand vertices/joints 与 object transform |
| HO-Cap | 完成 | 2/2 | 2/2 | PCA45、subject shape 与 qxyzw object pose |
| ContactPose | 完成 | 2/2 静态 | 2/2 strict | 静态单帧、官方 joints；论文 contact benchmark 尚未复现 |
| ARCTIC | TODO | — | — | 阶段 13 |
| OakInk2 | TODO | — | — | 阶段 13 |
| TACO | TODO | — | — | 阶段 13 |

## 机器人手支持

| 目标手 | 运动学 | 重定向 | 碰撞 | 仿真/RL |
|---|---|---|---|---|
| Arti-MANO | 已验证 | 已验证 | 已验证 | 未通过 RL qualification |
| Wuji Hand2 Beta1 | 已验证 | 已验证 | 已验证 | 仅离线 reference generation |
| 通用 URDF/MJCF | 导入基础 | 需要 manifest | 需要 profile | 不自动保证 |

## 环境配置

### 依赖与安装

- Linux、Git、Python `>=3.10,<3.14`；当前维护的本地 workflow 使用 Python 3.12。
- 完整流程需要 SciPy、PyTorch、Zarr、trimesh、SMPL-X 和 browser/visualization 依赖。
- 真实数据运行需要 GRAB 与 MANO 文件，并须遵守其上游许可证。

创建独立环境并安装所有已实现 workflow 的 extra：

```bash
conda create -n topo-retarget python=3.12 -y
conda activate topo-retarget
python -m pip install -U pip
python -m pip install -e ".[dev,cache,viz,grab,robot,geometry,retarget]"
```

### 配置本地资源

不要把有许可证约束的数据或 model 文件复制进 Git。可以直接设置路径，也可以依据
[configs/paths.example.yaml](configs/paths.example.yaml) 创建被忽略的 `.local/config.yaml`：

```bash
export GRAB_ROOT=/path/to/GRAB
# GRAB_ROOT 是 `toporetarget data index` 使用的 dataset root。
export MANO_MODEL_ROOT=/path/to/body_models/mano
# MANO_MODEL_ROOT 应包含 MANO_LEFT.pkl 与 MANO_RIGHT.pkl。

export PYTHONNOUSERSITE=1
export PYTHONPATH=src
export TOPORETARGET_PYTHON="${CONDA_PREFIX}/bin/python"
```

Arti-MANO 与 Wuji 资产已经 tracked 在 `third_party/robot_hands/`。Arti-MANO 可通过
`TOPORETARGET_ARTIMANO_ASSET_ROOT` 覆盖；未设置时使用 tracked bundle。

### 验证安装与资产

```bash
"$TOPORETARGET_PYTHON" -m toporetarget --help
"$TOPORETARGET_PYTHON" -m toporetarget doctor paper
"$TOPORETARGET_PYTHON" -m toporetarget robots list
"$TOPORETARGET_PYTHON" -m toporetarget robots validate artimano_rh \
  --asset-root third_party/robot_hands/artimano
"$TOPORETARGET_PYTHON" -m toporetarget robots validate wuji_hand2_beta1_rh \
  --asset-root third_party/robot_hands/wuji_hand2_beta1
```

## 完整 Workflow

以下主流程以命令为核心，不保存历史阶段日志。Raw GRAB 与 MANO 输入均为只读，生成的
artifact 保留在 `.local/`。

所有必要可视化统一使用 browser-based self-contained HTML。统一样式是全画布场景、右侧
控制栏、frame slider/playback、orbit/zoom、source/warm/final layer、graph/contact/collision
filter、metric 与 provenance，与
`.local/experiments/wuji_hand2_continuous_v1/html/W1_airplane_lift_continuity_comparison.html`
采用同类交互方式。PNG/GIF 与临时 Matplotlib window 不属于本 README 的 workflow。

### 1. 选择 source clip 与目标机器人手

`SEQUENCE`、`START_FRAME` 和 `END_FRAME` 共同选择 GRAB object/action 与 native
半开区间。下面示例使用右手 60 帧；可以改成其它已索引序列/区间，但不得 resample，也不得
依据结果重新选择。

```bash
# Source object/action：三个值应一起修改。
export SEQUENCE=s1/airplane_lift
export START_FRAME=240
export END_FRAME=300
export HAND=right

# 其它固定示例：
# apple:      SEQUENCE=s1/apple_eat_1      START_FRAME=212  END_FRAME=272
# alarmclock: SEQUENCE=s1/alarmclock_lift  START_FRAME=407  END_FRAME=467

# 目标手 family：artimano 或 wuji。
export TARGET_FAMILY=artimano

case "${TARGET_FAMILY}:${HAND}" in
  artimano:right)
    export ROBOT=artimano_rh
    export TARGET_ASSET_ROOT=third_party/robot_hands/artimano
    ;;
  artimano:left)
    export ROBOT=artimano_lh
    export TARGET_ASSET_ROOT=third_party/robot_hands/artimano
    ;;
  wuji:right)
    export ROBOT=wuji_hand2_beta1_rh
    export TARGET_ASSET_ROOT=third_party/robot_hands/wuji_hand2_beta1
    ;;
  wuji:left)
    export ROBOT=wuji_hand2_beta1_lh
    export TARGET_ASSET_ROOT=third_party/robot_hands/wuji_hand2_beta1
    ;;
  *)
    echo "unsupported TARGET_FAMILY/HAND pair" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

export GRAB_INDEX=.local/index/grab
export SOLVER_PROFILE=scipy_slsqp_active_set_contact_rich_v3_fixed
```

Arti-MANO 与 Wuji 共用数据/index 和目标手验证命令：

```bash
"$TOPORETARGET_PYTHON" -m toporetarget data index \
  --dataset grab --grab-root "$GRAB_ROOT" --output "$GRAB_INDEX"

"$TOPORETARGET_PYTHON" -m toporetarget data validate \
  --dataset grab --index "$GRAB_INDEX" --sequence "$SEQUENCE" \
  --hands "$HAND" --mano-model-root "$MANO_MODEL_ROOT" \
  --contact-mode semantic --start-frame "$START_FRAME" --end-frame "$END_FRAME" \
  --report .local/reports/preflight/source_validation.json

"$TOPORETARGET_PYTHON" -m toporetarget robots validate "$ROBOT" \
  --asset-root "$TARGET_ASSET_ROOT" \
  --report .local/reports/preflight/"${ROBOT}"_validation.json
```

### 2A. 运行完整 Arti-MANO pipeline

Manifest-driven Arti-MANO runner 会解析并验证 source、转换 canonical HOI/MANO 语义、
审计 object geometry、采样 object 与机器人 collision surface、生成并验证 warm start、
构建/评价冻结的 interaction graph、运行 final refinement、执行独立 collision/semantic
检查并生成 review bundle。
运行本 lane 前，应在第 1 节选择 `TARGET_FAMILY=artimano`。

Planning 不运行 solver：

```bash
test "$TARGET_FAMILY" = artimano
export RUN_ROOT=.local/runs/artimano
export WINDOW_LENGTH="$((END_FRAME - START_FRAME))"

"$TOPORETARGET_PYTHON" -m toporetarget workflow plan-grab \
  --sequence "$SEQUENCE" --index "$GRAB_INDEX" \
  --hand "$HAND" --robot "$ROBOT" \
  --start-frame "$START_FRAME" --end-frame "$END_FRAME" \
  --window-length "$WINDOW_LENGTH" \
  --refinement-solver-profile "$SOLVER_PROFILE" \
  --mano-model-root "$MANO_MODEL_ROOT" --run-root "$RUN_ROOT" \
  --output .local/reports/preflight/workflow_plan.json --dry-run
```

运行或恢复完整 DAG：

```bash
"$TOPORETARGET_PYTHON" -m toporetarget workflow run-grab \
  --sequence "$SEQUENCE" --index "$GRAB_INDEX" \
  --hand "$HAND" --robot "$ROBOT" \
  --start-frame "$START_FRAME" --end-frame "$END_FRAME" \
  --window-length "$WINDOW_LENGTH" \
  --refinement-solver-profile "$SOLVER_PROFILE" \
  --mano-model-root "$MANO_MODEL_ROOT" --asset-root "$TARGET_ASSET_ROOT" \
  --run-root "$RUN_ROOT" --resume

RUN_ID="${SEQUENCE//\//__}__${HAND}__${ROBOT}__f$(printf '%06d' "$START_FRAME")_f$(printf '%06d' "$END_FRAME")"
export RUN_DIR="$RUN_ROOT/$RUN_ID"
export RUN_MANIFEST="$RUN_DIR/manifest.json"
```

`workflow run-grab` 当前按设计只验证 Arti-MANO target name。Wuji 应使用下一节的 suite
runner，不要把 Wuji robot name 直接替换进本命令。

### 2B. 运行完整 Wuji pipeline

通用 suite runner 对冻结的 Wuji clip 运行相同的 canonical conversion、geometry、warm-start、
interaction-graph、refinement、validation、export 与 HTML evaluation component。使用
`--unit` 选择一个 object；省略则运行右手 W1/W2/W3。

```bash
export TARGET_FAMILY=wuji
export HAND=right
export ROBOT=wuji_hand2_beta1_rh
export TARGET_ASSET_ROOT=third_party/robot_hands/wuji_hand2_beta1
export WUJI_EXPERIMENT_ROOT=.local/experiments/wuji_hand2_grab3_v1

"$TOPORETARGET_PYTHON" -m toporetarget workflow run-grab-suite \
  --suite configs/experiments/wuji_hand2_grab3_v1.yaml \
  --grab-root "$GRAB_ROOT" --index "$GRAB_INDEX" \
  --mano-model-root "$MANO_MODEL_ROOT" \
  --robot "$ROBOT" --solver-profile "$SOLVER_PROFILE" \
  --experiment-root "$WUJI_EXPERIMENT_ROOT" \
  --resume --max-wall-time 1800 \
  --evaluate --export-reference --generate-html

# 仅运行所选 airplane 示例时追加：
# --unit W1_s1__airplane_lift__right__wuji_hand2_beta1_rh__f000240_f000300
```

Wuji suite 的权威 machine status 位于
`$WUJI_EXPERIMENT_ROOT/reports/final_status.json`，export 位于 `exports/`，HTML 入口位于
`html/index.html`。

### 3. 验证、审计并导出 Arti-MANO run

以下检查由 manifest 驱动，不修改 raw data。JSON/CSV 是权威结果，HTML 是视觉审计界面。

```bash
"$TOPORETARGET_PYTHON" -m toporetarget workflow status --run "$RUN_MANIFEST"

"$TOPORETARGET_PYTHON" -m toporetarget workflow validate \
  --run "$RUN_MANIFEST" \
  --report "$RUN_DIR/reports/end_to_end_validation.json" \
  --csv "$RUN_DIR/reports/end_to_end_validation.csv"

"$TOPORETARGET_PYTHON" -m toporetarget workflow audit-contact-retention \
  --run "$RUN_MANIFEST" \
  --output-dir "$RUN_DIR/audits/contact_retention" \
  --surface-samples 8192 --thresholds-mm 1,2,3,5,8,10 \
  --html --force

"$TOPORETARGET_PYTHON" -m toporetarget workflow export-reference \
  --run "$RUN_MANIFEST" --format zarr \
  --output "$RUN_DIR/exports/robot_reference.zarr"
```

导出的 `toporetarget.robot_reference.v1` 是离线 trajectory artifact，不是真实机器人控制指令。

### 4. 生成统一的中间/最终 HTML review

一个 combined page 覆盖必要的中间与最终视图：source MANO mesh、warm-start target mesh、
final target mesh、object context、冻结的 interaction graph、Figure-4-style hand-object edge、
Laplacian residual、逐帧 refinement metric 与 provenance。

```bash
"$TOPORETARGET_PYTHON" -m toporetarget workflow visualize-mesh \
  --run "$RUN_MANIFEST" --mode combined \
  --output "$RUN_DIR/review/trajectory_comparison.html" \
  --interactive
```

Contact audit 会生成第二个 self-contained 页面：
`$RUN_DIR/audits/contact_retention/trajectory_contact_audit.html`。它提供
source/warm/final、visual/collision surface、QuerySet、semantic anchor、threshold、frame
与 link/region 控制。视觉合理不能替代 numerical validation 与 collision report。

Wuji 的 `--generate-html` 会在 `$WUJI_EXPERIMENT_ROOT/html/` 下生成相应的全画布页面。
打开 `index.html`；每个 clip 页面都提供 source/warm/final layer 及同类 browser playback/control。

### 5. 完成人工 review 边界

Machine validation 不能伪造人工 acceptance。先生成 template，在 combined/contact-audit HTML
中检查 required 与 worst frame，再由具名人工 reviewer 填写复制后的 record：

```bash
"$TOPORETARGET_PYTHON" -m toporetarget workflow review-template \
  --run "$RUN_MANIFEST" --output "$RUN_DIR/review/manual_acceptance.template.json"

cp "$RUN_DIR/review/manual_acceptance.template.json" \
  "$RUN_DIR/review/manual_acceptance.json"
# 由人工 reviewer 填写 manual_acceptance.json；不得自动写入 pass。
```

使用以下参数恢复第 2A 节同一条 `workflow run-grab` 命令：

```text
--manual-acceptance "$RUN_DIR/review/manual_acceptance.json"
```

Content hash、selected frame range、source identity、robot/profile identity、solver status、
collision/continuity gate 与人工 review lineage 必须同时有效。失败 gate 应保持失败；不得跳帧、
替换 object 或根据结果重新选择。

### 6. 复现已实现的 evaluation lane

这些是独立的冻结评价，不是 single-run workflow 中的隐藏步骤。

Arti-MANO 四 clip quality/morphology/contact evaluation：

```bash
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
"$TOPORETARGET_PYTHON" -m toporetarget quality run-a-to-e \
  --config configs/experiments/grab_artimano_quality_v1.yaml \
  --resume --max-wall-time 1800 --generate-html

"$TOPORETARGET_PYTHON" -m toporetarget quality status \
  --experiment-root .local/experiments/grab_artimano_quality_v1
# HTML：.local/experiments/grab_artimano_quality_v1/html/index.html
```

Wuji continuity evaluation 与 comparison HTML：

```bash
"$TOPORETARGET_PYTHON" -m toporetarget workflow run-grab-suite \
  --suite configs/experiments/wuji_hand2_grab3_v1.yaml \
  --grab-root "$GRAB_ROOT" --index "$GRAB_INDEX" \
  --mano-model-root "$MANO_MODEL_ROOT" \
  --robot wuji_hand2_beta1_rh \
  --solver-profile wuji_continuous_full_state_v1 \
  --experiment-root .local/experiments/wuji_hand2_continuous_v1 \
  --resume --max-wall-time 1800 \
  --evaluate --export-reference --generate-html
# HTML：.local/experiments/wuji_hand2_continuous_v1/html/index.html
```

冻结的 GRAB/ContactPose benchmark 与统一 dashboard：

```bash
export CONTACTPOSE_ROOT=/path/to/ContactPose

"$TOPORETARGET_PYTHON" -m toporetarget benchmark inspect-datasets \
  --grab-root "$GRAB_ROOT" --contactpose-root "$CONTACTPOSE_ROOT" \
  --output .local/benchmarks/hoi_benchmark_v1/dataset_audit.json
"$TOPORETARGET_PYTHON" -m toporetarget benchmark select \
  --config configs/benchmarks/hoi_benchmark_v1.yaml
"$TOPORETARGET_PYTHON" -m toporetarget benchmark freeze
"$TOPORETARGET_PYTHON" -m toporetarget benchmark run --resume
"$TOPORETARGET_PYTHON" -m toporetarget benchmark evaluate --html
"$TOPORETARGET_PYTHON" -m toporetarget benchmark dashboard
```

Selection 与 attribution gate 采用 fail closed。Static ContactPose unit 与 dynamic GRAB unit
保持分离，GRAB contact proxy 绝不重命名为 ContactPose ground truth。

### 7. 仓库级审计

```bash
"$TOPORETARGET_PYTHON" scripts/check_paper_fidelity.py
"$TOPORETARGET_PYTHON" -m pytest -m "not licensed_data"
"$TOPORETARGET_PYTHON" -m ruff check .
"$TOPORETARGET_PYTHON" -m ruff format --check .
"$TOPORETARGET_PYTHON" -m mypy src
git diff --check
```

Licensed-data test 是 opt-in，要求已配置本地 GRAB/MANO 资源。

## 其它文档索引

- 项目规划与历史：
  [路线图](docs/ROADMAP.md) /
  [中文路线图](docs/ROADMAP.zh-CN.md)、
  [开发日志](docs/DEVELOPMENT_LOG.md) /
  [中文开发日志](docs/DEVELOPMENT_LOG.zh-CN.md)、
  [复现日志](docs/REPRODUCTION_LOG.md)
- 论文与契约：
  [论文忠实度](docs/PAPER_FIDELITY.md)、
  [实现规范](docs/PAPER_IMPLEMENTATION_SPEC.md)、
  [显式假设](docs/ASSUMPTIONS.md)、
  [待向作者确认的问题](docs/OPEN_QUESTIONS_FOR_AUTHORS.md)
- 数据与 geometry：
  [数据布局](docs/DATA_LAYOUT.md)、
  [GRAB adapter](docs/GRAB_DATASET_ADAPTER.md)、
  [MANO 转换](docs/MANO_TO_MEDIAPIPE21.md)、
  [Object geometry/sampling](docs/OBJECT_GEOMETRY_AND_SAMPLING.md)、
  [Signed distance](docs/SIGNED_DISTANCE_AND_COLLISION_QUERIES.md)
- 重定向：
  [Relative-bone 初始化](docs/RELATIVE_BONE_DIRECTION_INITIALIZATION.md)、
  [Interaction graph](docs/INTERACTION_GRAPH.md)、
  [Laplacian loss](docs/LAPLACIAN_INTERACTION_LOSS.md)、
  [Final refinement](docs/CONTACT_PRESERVING_FINAL_REFINEMENT.md)、
  [Workflow resume/provenance](docs/WORKFLOW_RESUME_AND_PROVENANCE.md)
- HTML review 与审计：
  [Trajectory visualization](docs/TRAJECTORY_VISUALIZATION.md)、
  [Interaction-mesh HTML](docs/INTERACTION_MESH_VISUALIZATION.md)、
  [Contact-retention audit](docs/CONTACT_RETENTION_AUDIT.md)、
  [Warm-start audit](docs/WARM_START_FIDELITY_AND_REACHABILITY_AUDIT.md)
- 目标手与评价：
  [Tracked robot assets](docs/TRACKED_ROBOT_HAND_ASSETS.md)、
  [Arti-MANO adapter](docs/ARTIMANO_ADAPTER.md)、
  [Wuji target](docs/WUJI_HAND2_BETA1_TARGET.md) /
  [中文 Wuji 目标手](docs/WUJI_HAND2_BETA1_TARGET.zh-CN.md)、
  [Arti-MANO A–E 评价](docs/GRAB_ARTIMANO_QUALITY_EXPERIMENT.md)、
  [Wuji GRAB 重定向](docs/WUJI_HAND2_GRAB_RETARGETING.md)、
  [Wuji continuity](docs/WUJI_CONTINUOUS_RETARGETING.md)
- 仓库策略：
  [数据/许可证策略](docs/LICENSE_AND_DATA_POLICY.md)、
  [第三方资产策略](docs/THIRD_PARTY_ASSET_POLICY.md)、
  [贡献指南](CONTRIBUTING.md)、
  [第三方声明](THIRD_PARTY_NOTICES.md)

详细阶段历史和实现记录维护在：

- [docs/DEVELOPMENT_LOG.zh-CN.md](docs/DEVELOPMENT_LOG.zh-CN.md)
- [docs/ROADMAP.zh-CN.md](docs/ROADMAP.zh-CN.md)
- [docs/stages/](docs/stages/)
- [solver-feasibility 说明](docs/SOLVER_FEASIBILITY_RESTORATION.md)
- [Stage 16 reference-tracking PPO](docs/stages/STAGE16_REFERENCE_TRACKING_PPO.md)

## License

仓库代码与文档采用 GNU General Public License v3.0，见 [LICENSE](LICENSE)。Tracked
第三方资产继续使用其上游许可证与 `third_party/robot_hands/` 中的 notice。GRAB、
MANO/SMPL-X、ContactPose、ManipTrans 及其它外部资源受各自条款约束。使用前请阅读
[docs/LICENSE_AND_DATA_POLICY.md](docs/LICENSE_AND_DATA_POLICY.md)和
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## Acknowledgments

感谢：

- [TopoRetarget](https://toporetarget2026.github.io/TopoRetarget/) 作者；
- GRAB、MANO/SMPL-X、ContactPose 的作者与维护者；
- ManipTrans 项目与 Arti-MANO 上游资产贡献者；
- tracked provenance manifest 中记录的 Wuji Hand2 上游资产贡献者。

使用时请保留上游 attribution，并遵守每个 dataset、model、code 与 asset 的许可证。

## Citation

如果本仓库或实现记录对你的工作有帮助，请引用 TopoRetarget 论文：

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

同时请引用实验实际使用的 dataset、body model、目标手资产和上游实现。本地论文副本见
[docs/TopoRetarget.pdf](docs/TopoRetarget.pdf)，上游引用索引见
[docs/UPSTREAM_REFERENCES.md](docs/UPSTREAM_REFERENCES.md)。
