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

### H3 物理准入与未见物体 claim 边界

`Executable V2 是物理 admission hard gate`：它判断 source controller 是否有限、
有界、安全且可执行。`Fidelity V2 是诊断/warm-start 质量`：task/contact imitation
退化会继续进入冻结的 full-gravity evaluation，而不会被重新标成执行失败。真实 joint、
actuator、collision、velocity、effort 和 action limits 均保留。

冻结的 H3 Hardening5 回归得到五条数值 exact-retarget 终态，但后续
`RetargetSemanticValidityV1` 审计发现：原 HOCap 路径把 MANO 参数 frame 当成了语义 wrist，
所以旧 reference 均未通过语义资格化。三条历史 PPO failure 必须降级为
`NON_DIAGNOSTIC_INVALID_REFERENCE`；重新生成的几何 reference 不会追溯改变这些不可变 trace。
另外两条进入明确的 `SUPPORT_UNRESOLVED` 物理无效状态。因此
`H3C_READY_FOR_UNSEEN_OBJECT_EXECUTION=NO`。object/mesh-disjoint Frozen5 已冻结并完成
审计，但没有消费其 downstream Episode。本轮`不做 shared-policy zero-shot claim`；合同要求
`每条 Episode 独立 PPO`，且本轮不声明未见物体性能结论。

### Dataset semantic authority 与双 canary gate（P0-P5）

在任何 exact retarget 或 physicalization admission 之前，先对完整 HOCap corpus 执行只读的
`DatasetSemanticAuthorityV1` audit。它解析所有官方 hand slot 和 object candidate，记录
`CanonicalHOIRecordV1` authority chain，并对 target object 歧义、object-asset binding 错误、生命周期不完整、
bimanual same-object episode、frame/time authority 缺陷 fail closed。P0-P4 artifacts 写入一个 ignored report root，
并使用 `TWO_CANARY_SELECTION_SEED=20260830` 冻结恰好两条新的 right-hand semantic-PASS canary：

```bash
conda run --no-capture-output -n topo-retarget python \
  scripts/evaluation/run_dataset_semantic_authority.py \
  --episode-index <all_hocap_episodes.json> \
  --data-root <HOCap-root> \
  --output-root <report-root>/dataset_semantic_authority_two_clip_canary \
  --force
```

只有冻结 manifest 中的两条 entry 可以进入
`wuji_continuous_sequential_fast_exact_v2`。对两条 receipt-bound HTML 进行人工检查，并分别执行
`qualify_retarget_semantics.py`。第一次执行在
`WAITING_FOR_USER_RETARGET_HTML_ACCEPTANCE` 停止；不会启动 PPO、support、PhysX、reward、RSE、PF、DF 或任何
P6-P8 route。恢复必须对每条 canary 分别给出明确决定：
`CANARY_1=APPROVE` 与 `CANARY_2=APPROVE`。

## 方法总览

```text
有授权的 HOI 数据
  -> 规范 HOI 序列与坐标约定
  -> MANO / 目标手语义转换
  -> 交互感知运动学重定向
  -> RetargetSemanticValidityV1 的 frame、时间、几何、接触与连续性验证
  -> source/canonical/warm/final HTML 人工复核
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
因果物理 PPO 精炼、评价和 replay。授权的原始数据保持只读；派生 cache、report 和 HTML 请写到仓库外，例如
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

### 4. 人类 HOCap Episode 到物理机器人示范

当前权威为
[`HOCapPhysicalizationHardeningProtocolV2`](configs/contracts/hocap_physicalization_hardening_v2.json)。
production unit 是一条从 approach、pick、place、release 到 retreat 的完整
`HOCapSingleHandObjectEpisodeV1`；raw-sequence/primary-object window 仅为历史诊断。
HOCap/MANO 原始输入只读。先设置彼此独立的输出目录，
并在执行前检查各入口的 `--help`：

```bash
export HOCAP_ROOT=/path/to/HOCap
export MANO_MODEL_ROOT=/path/to/mano
export EPISODE_ROOT=/path/to/reports/episodes
export PHYS_RUN_ROOT=/path/to/runs/physicalization_v2
export PHYS_REPORT_ROOT=/path/to/reports/physicalization_v2
export EPISODE_ID=<frozen-episode-id>
```

1. 解析 raw sequence。`auto` 会解析 HOCap 官方的两个 hand slot；target object 由
   whole-MANO-surface 到精确 object triangle mesh 的完整生命周期证据确定。`--hand left/right`
   只用于显式诊断过滤。

   ```bash
   conda run -n topo-retarget python scripts/data/parse_hocap_episodes.py \
     --data-root "$HOCAP_ROOT" --mano-model-root "$MANO_MODEL_ROOT" \
     --output-root "$EPISODE_ROOT" --hand auto --resume
   ```

2. 人工检查 active hand、target object，以及 approach、pickup、place、release、retreat：

   ```bash
   conda run -n topo-retarget python scripts/visualize_hocap_episode.py \
     --episode-index "$EPISODE_ROOT/all_hocap_episodes.json" \
     --episode-id "$EPISODE_ID" --data-root "$HOCAP_ROOT" \
     --mano-model-root "$MANO_MODEL_ROOT" \
     --output "$PHYS_REPORT_ROOT/episode.html" \
     --sanity-output "$PHYS_REPORT_ROOT/episode_sanity.json"
   ```

3. 在昂贵 solver 前执行 `RetargetInputQualityV1`。输入被拒绝时停止该 episode；PASS receipt
   会绑定原输入或只修复短 gap 后的输入。

   ```bash
   conda run -n topo-retarget python scripts/retarget/scan_hocap_retarget_input_quality.py \
     --episode-index "$EPISODE_ROOT/all_hocap_episodes.json" \
     --episode-id "$EPISODE_ID" --data-root "$HOCAP_ROOT" \
     --mano-model-root "$MANO_MODEL_ROOT" \
     --report "$PHYS_REPORT_ROOT/retarget_input_quality.json" \
     --per-frame-csv "$PHYS_REPORT_ROOT/retarget_input_quality_per_frame.csv" \
     --repaired-output "$PHYS_RUN_ROOT/retarget_input_quality_repaired.npz"
   ```

4. 使用数学不变的 `fast_exact_v2` execution profile 执行几何重定向。production 不使用
   `--benchmark-first-frames` 或 `--skip-html`。

   ```bash
   conda run -n topo-retarget python scripts/run_hocap_episode_geometric_retarget.py \
     --episode-index "$EPISODE_ROOT/all_hocap_episodes.json" \
     --episode-id "$EPISODE_ID" --data-root "$HOCAP_ROOT" \
     --mano-model-root "$MANO_MODEL_ROOT" \
     --selection-manifest <frozen-episode-manifest.json> \
     --execution-profile wuji_continuous_sequential_fast_exact_v2 \
     --run-root "$PHYS_RUN_ROOT/geometric" \
     --report-root "$PHYS_REPORT_ROOT/geometric"
   ```

5. 在生成任何 physical reference 前，对 receipt-bound source、warm 和 final artifact
   执行语义资格化。**优化器数值收敛不等价于几何重定向语义正确。** production admission 必须同时满足
   `NumericalSolverSuccess` 与 `RetargetSemanticValidityV1`；`FAIL` 或 `INCONCLUSIVE` 均停止
   downstream。第一个命令是单 Episode 的 fail-closed production gate；随后注册审计命令
   会先重新计算不可变 170105/170650 正控对比，再检查 Hardening5。

   ```bash
   GEOMETRIC_RUN="$PHYS_RUN_ROOT/geometric/$EPISODE_ID"
   GEOMETRIC_REPORT="$PHYS_REPORT_ROOT/geometric/episodes/$EPISODE_ID"
   conda run -n toporetarget-rl python \
     scripts/evaluation/qualify_retarget_semantics.py \
     --episode-id "$EPISODE_ID" \
     --canonical "$GEOMETRIC_RUN/raw_contract/canonical_episode.zarr" \
     --warm-start "$GEOMETRIC_RUN/retarget/warm_start.npz" \
     --final "$GEOMETRIC_RUN/retarget/final_continuous.zarr" \
     --graph "$GEOMETRIC_RUN/retarget/interaction_graph.npz" \
     --evaluation "$GEOMETRIC_RUN/retarget/interaction_evaluation.npz" \
     --viewer "$GEOMETRIC_REPORT/retarget/continuous_refinement_visualization.html" \
     --receipt "$GEOMETRIC_REPORT/geometric_retarget_receipt.json" \
     --output "$GEOMETRIC_REPORT/retarget/semantic_qualification.json" \
     --per-frame-csv "$GEOMETRIC_REPORT/retarget/semantic_metrics_per_frame.csv"

   export SEMANTIC_REPORT_ROOT="$PHYS_REPORT_ROOT/retarget_semantic_validity_frame_authority_audit"
   conda run -n toporetarget-rl python \
     scripts/evaluation/audit_retarget_semantic_validity.py \
     --phase post_fix --output-root "$SEMANTIC_REPORT_ROOT" \
     --episode-index "$EPISODE_ROOT/all_hocap_episodes.json" \
     --positive-control-root <accepted-stage12-hocap-root> \
     --hardening-run-root <hardening5-geometric-run-root> \
     --hardening-report-root <hardening5-geometric-report-root>
   column -s, -t "$SEMANTIC_REPORT_ROOT/positive_controls/comparison.csv"
   column -s, -t "$SEMANTIC_REPORT_ROOT/hardening5/main_table.csv"
   ```

6. 打开输出的 `continuous_refinement_visualization.html`。统一 semantic viewer 支持
   RAW/CANONICAL/WARM/FINAL/object 切换、显式 frame axes、fingertips、冻结 interaction
   graph 与 warm→final residual vectors。若要从相同、receipt-bound artifact 重新生成：

   ```bash
   conda run -n topo-retarget python -m toporetarget workflow visualize-mesh \
     --run <html_visualization_manifest.json> --mode combined \
     --max-object-points 50000 --output <retarget.html>
   ```

7. 从通过验证的 final trajectory 与 checkpoint manifest 构建 physical reference：

   ```bash
   conda run -n toporetarget-rl python scripts/rl/prepare_independent_source_reference.py \
     --clip-id "$EPISODE_ID" --final-trajectory <final_continuous.zarr> \
     --canonical <canonical_episode.zarr> \
     --geometric-receipt "$GEOMETRIC_REPORT/geometric_retarget_receipt.json" \
     --semantic-qualification "$GEOMETRIC_REPORT/retarget/semantic_qualification.json" \
     --checkpoint-manifest <continuous_checkpoints/manifest.json> \
     --wuji-mjcf third_party/robot_hands/wuji_hand2_beta1/mjcf/right.xml \
     --world-reference-output <world_reference.npz> --object-mesh-output <object.obj> \
     --reference-v1-output <reference_v1.npz> \
     --reference-v2-output <reference_kinematics_v2.npz> --report <reference.json>
   ```

8. 在精确 Isaac 环境中冻结 host GPU authority。sandbox 中 CUDA 失败只是一条诊断，
   不等价于 host GPU 不可用；禁止 CPU fallback。

   ```bash
   nvidia-smi -L
   conda run --no-capture-output -n toporetarget-isaaclab \
     python scripts/runtime/gpu_preflight.py \
     --execution-context host-unsandboxed --isaac-bootstrap --accept-eula \
     --output <gpu_preflight_receipt.json>
   ```

9. 先运行 zero-residual deterministic source controller，并使用 continuous equivalent-angle
   virtual wrist 与真实 finger limits。L0 是条件 fallback，不是每条 episode 自动必需。

   ```bash
   conda run --no-capture-output -n toporetarget-isaaclab \
     python scripts/rl/isaaclab/qualify_zero_residual_source_controller.py \
     --accept-eula --clip "$EPISODE_ID" --episodes 10 \
     --output "$PHYS_REPORT_ROOT/source_controller/zero_residual" \
     --reference <reference_kinematics_v2.npz> --object-usd <object.usda> \
     --support-proxy <table_proxy.json> --support-asset <support_proxy.usda> \
     --contact-contract <contact_contract.json> --contact-mask-root <contact_mask_root> \
     --reference-distance-root <reference_distance_root> \
     --object-mesh-root <object_mesh_root> \
     --runtime-geometry-manifest <runtime_collision_geometry_manifest.json> \
     --frozen-evaluation-gates <frozen_evaluation_gates.json> \
     --seed-manifest <seed_manifest.json>
   ```

10. 仅当步骤 9 FAIL 时，训练恰好 `1,024,000` samples 的 corrected L0 actor，并用相同
   Eval10 qualification。`--continuous-virtual-wrist-angles` 只消除表示 wrapping failure，
   不移除真实 finger、action、effort、velocity、singularity、collision 或 actuator limit。

   ```bash
   conda run --no-capture-output -n toporetarget-isaaclab \
     python scripts/rl/isaaclab/train_stage16d_ppo26d.py --accept-eula \
     --clip "$EPISODE_ID" --reference <reference_kinematics_v2.npz> \
     --object-usd <object.usda> --output-root "$PHYS_RUN_ROOT/source_controller/corrected_l0" \
     --num-envs 1024 --iterations 25 --seed <frozen-seed> \
     --continuous-virtual-wrist-angles
   conda run --no-capture-output -n toporetarget-isaaclab \
     python scripts/rl/isaaclab/qualify_zero_residual_source_controller.py \
     --accept-eula --clip "$EPISODE_ID" --episodes 10 \
     --checkpoint <corrected_l0_checkpoint.pt> --optimizer-steps 25 \
     --training-samples 1024000 \
     --output "$PHYS_REPORT_ROOT/source_controller/corrected_l0" \
     --reference <reference_kinematics_v2.npz> --object-usd <object.usda> \
     --support-proxy <table_proxy.json> --support-asset <support_proxy.usda> \
     --contact-contract <contact_contract.json> --contact-mask-root <contact_mask_root> \
     --reference-distance-root <reference_distance_root> \
     --object-mesh-root <object_mesh_root> \
     --runtime-geometry-manifest <runtime_collision_geometry_manifest.json> \
     --frozen-evaluation-gates <frozen_evaluation_gates.json> \
     --seed-manifest <seed_manifest.json>
   ```

11. 在进入仿真前应用 V2 physical-scene authority。`ObjectDynamicsAuthorityV1`
   记录 mass、COM、inertia 是 explicit、derived 还是 unresolved；缺少 source
   support geometry 只能记为 `SUPPORT_UNDERDETERMINED`，不能推断为
   `SUPPORT_ABSENT`。`SupportExistenceContractV1` 与
   `SupportPhysicalizationV1` 是两个独立决策。按冻结优先级
   `SUPPORT_ONLY -> COMMON_SCENE_SE3 -> RELATIVE_OBJECT_PROJECTION` 解析；前两者
   保持手-物相对变换，因此复用已验证的几何重定向。`RELATIVE_OBJECT_PROJECTION`
   只有在 exact retarget 与 semantic revalidation 通过、并经过 hard human gate
   后才允许。必须在看到 canary 结果前冻结一个全局
   `PhysicalizationDeviationBudgetV1`；禁止按 canary 调 friction、mass、COM、inertia、
   reward、PPO、wrist、finger 或 trajectory。

   settled-support qualification 使用 runtime `dt` 与秒域 terminal window；impact peak
   只作诊断。只有在 `SettledSupportDynamicsQualificationV2` 下 contact、terminal motion
   有界、runtime COM/inertia provenance 均通过，scene 才算 ready。

   ```text
   source semantics -> support existence -> physicalization -> settled dynamics
                                  \-> frozen retarget reuse decision
   ```

   若 source table 参数存在必须恢复，不得再推断第二张 table。source/reconstructed
   support 的 hand/object collision 都为 ON；inferred proxy 仅用 pairwise filter 将
   hand/support collision 设为 OFF，object collision 保持 ON。

   ```bash
   conda run -n topo-retarget python scripts/physics/run_independent_physical_support.py \
     --manifest <frozen-episode-manifest.json> --clip-id "$EPISODE_ID" \
     --source-policy-receipt <source_policy_receipt.v3.json> \
     --run-root "$PHYS_RUN_ROOT" --report-root "$PHYS_REPORT_ROOT" \
     --base-runtime-geometry-manifest <runtime_collision_geometry_manifest.json> \
     --gpu-preflight-receipt <gpu_preflight_receipt.json> --accept-eula
   ```

12. 在任何 physical update 前执行冻结的 full-gravity Eval10：

   ```bash
   conda run -n topo-retarget python scripts/evaluation/run_independent_frozen_physical_evaluation.py \
     --manifest <frozen-episode-manifest.json> --clip-id "$EPISODE_ID" \
     --source-policy-receipt <source_policy_receipt.v3.json> \
     --support-receipt <support_receipt.json> \
     --gpu-preflight-receipt <gpu_preflight_receipt.json> \
     --interaction-contact-contract <interaction_contact_contract.json> \
     --run-root "$PHYS_RUN_ROOT" --report-root "$PHYS_REPORT_ROOT" --accept-eula
   ```

13. 只按 PF V2 决策：PASS 以 0 次 PPO update 接受 frozen policy；只有 FAIL 才授权
    physical PPO。获得授权后按顺序执行三个 fail-closed mode。V2 的 P5 冻结 fallback 最多
    15 updates（`614,400` samples），并明确标记 `LENGTH_GENERALIZATION_NOT_ESTABLISHED`。
    RSI 为 `0.5*U(T_valid)+0.5*U(EpisodeV1 CONTACT through RELEASE)`，保留 uniform component；
    Confirm20 接受后提前停止。

    ```bash
    PPO_ARGS=(
      --clip "$EPISODE_ID" --num-envs 1024 --max-new-updates 15 --accept-eula
      --report-root "$PHYS_REPORT_ROOT/ppo" --run-root "$PHYS_RUN_ROOT/ppo"
      --source-training-result <l0_training.json> --reference <reference_v2.npz>
      --object-usd <object.usda> --support-proxy <table_proxy.json>
      --support-asset <support_proxy.usda> --contact-contract <contact_contract.json>
      --contact-mask-root <contact_mask_root> --reference-distance-root <reference_distance_root>
      --object-mesh-root <object_mesh_root>
      --runtime-geometry-manifest <runtime_collision_geometry_manifest.json>
      --frozen-evaluation-gates <frozen_evaluation_gates.json>
      --seed-manifest <seed_manifest.json>
      --hardening-v2-runtime-events <hardening_v2_runtime_events.json>
      --continuous-virtual-wrist-angles
      --gpu-preflight-receipt <gpu_preflight_receipt.json>
    )
    conda run --no-capture-output -n toporetarget-isaaclab \
      python scripts/rl/isaaclab/run_physical_refinement.py evaluate-first "${PPO_ARGS[@]}"
    conda run --no-capture-output -n toporetarget-isaaclab \
      python scripts/rl/isaaclab/run_physical_refinement.py runtime-sanity "${PPO_ARGS[@]}"
    conda run --no-capture-output -n toporetarget-isaaclab \
      python scripts/rl/isaaclab/run_physical_refinement.py train "${PPO_ARGS[@]}"
    ```

14. 对不可变 trace 执行 `PhysicalFunctionalityFullCycleV1`。PF V2 仍只负责 pick/lift；
    FullCycle V1 分别测量 pick、transport、place、release、retreat。没有记录 destination-region
    或 destination-support signal 时必须报 `NOT_IDENTIFIABLE`，不得用 source-table contact 代替。

    ```bash
    PYTHONPATH=src:. conda run -n topo-retarget \
      python scripts/evaluation/qualify_physical_functionality_full_cycle.py \
      --trace-root <qualification_dir/traces> \
      --runtime-events <hardening_v2_runtime_events.json> \
      --output <qualification_dir/full_cycle> --geometry-safe
    ```

15. replay 使用不可变 trace。相同入口支持完整 trajectory、window、raw MANO/object overlay、
    reference 开关和确定性的 low-poly raw object。

    ```bash
    conda run --no-capture-output -n toporetarget-isaaclab \
      python scripts/rl/isaaclab/replay_physical_hoi_trace.py --accept-eula \
      --trace <episode_000.npz> --object "$EPISODE_ID" --loop
    conda run --no-capture-output -n toporetarget-isaaclab \
      python scripts/rl/isaaclab/replay_physical_hoi_trace.py --accept-eula \
      --trace <episode_000.npz> --object "$EPISODE_ID" \
      --start-frame <start> --end-frame <end> --no-reference-ghost \
      --mocap-ghost --mocap-object-low-poly --loop
    ```

PF V2 测量 pick/lift；PF FullCycle V1 测量完整 manipulation。DF pose、linear、angular
保持独立，interaction timing 仅为诊断。replay 不训练 PPO，也不生成科学验收。

<details>
<summary>历史 two-clip 开发记录（不是当前 authority）</summary>

下列材料只为保留 provenance。不得用它选择 production unit 或冻结新 held-out manifest。

该 production workflow 从已验证的几何重定向输出开始，到已接受的 physical-HOI trace 结束。当前证据
仅覆盖物理 runner 支持的两条 HOCap clip。旧开发记录可能称其为 Stage16-D；production 命令不使用该名称。

1. 先准备 source-first support。若 source 没有可恢复的 support，则使用
   `INFERRED_PLANAR_SUPPORT`。有限 table 在整个 episode 始终 active，不能在接触后关闭。

   ```bash
   PYTHONPATH=src python scripts/physics/prepare_physical_support.py \
     --dataset hocap --sequence <clip> --support auto \
     --output-root <support_output> --static
   ```

2. 冻结的 full-gravity evaluation 使用 C4 physics：nominal friction、object gravity on、
   hand/virtual wrist gravity off、active support，且没有 guidance、attachment、rollout-time
   object-state write 或 wrist-root write。

   ```bash
   conda run -n toporetarget-isaaclab env OMNI_KIT_ACCEPT_EULA=YES \
     python scripts/evaluation/qualify_physical_hoi.py --accept-eula \
     --clip <clip> --checkpoint <frozen_checkpoint> \
     --output <run_dir>/eval10 --episodes 10 --update 0 --samples 0
   ```

3. 阅读 PF V2（当前 physical functionality authority）。它要求 physical lift、causal
   hand-object lift、support transfer、sustained coupling、geometry safety 和 no-cheating
   contract。PF V1 仍可查询，但只是 legacy timing-constrained metric；
   pre-reference-LIFT persistent multi-contact 现在只是 interaction timing diagnostic，不是 PF V2 hard gate。

4. PPO 前必须决策。PF V2 PASS 则接受 frozen policy，PPO update 为 0；PF V2 FAIL 才允许在
   evaluate-first receipt 与 no-step sanity gate 后做 bounded refinement。

   ```bash
   conda run -n toporetarget-isaaclab env OMNI_KIT_ACCEPT_EULA=YES \
     python scripts/rl/isaaclab/run_physical_refinement.py evaluate-first \
     --clip <clip> --accept-eula
   conda run -n toporetarget-isaaclab env OMNI_KIT_ACCEPT_EULA=YES \
     python scripts/rl/isaaclab/run_physical_refinement.py runtime-sanity \
     --clip <clip> --accept-eula
   conda run -n toporetarget-isaaclab env OMNI_KIT_ACCEPT_EULA=YES \
     python scripts/rl/isaaclab/run_physical_refinement.py train \
     --clip <clip> --accept-eula
   ```

   `train` 会再次执行 evaluate-first。candidate 达到 PF V2 Eval10 后运行 Confirm20；Confirm20
   acceptance 会持久化 checkpoint 并立即停止。`configs/rl/physical_refinement.yaml` 的
   `max_new_updates=10` 是上限，不是必须跑满的目标。

精炼 reward 为 grouped multiplicative：`R = R_obj * R_hand * R_int * R_reg`。group 内先聚合，
group 间再相乘，因此任一弱 group 都不能被另一 group 抵消。`R_int` 混合未改变的 V4 contact 与
geometric proximity。RSE 保持 training RSI 在 `[0,320]` uniform 采样，evaluation 则从 frame 0
确定性运行完整 trajectory。其冻结的全局项为 `w_scope(D_ref)=clip(D_ref/0.20,0,1)` 和
`kappa=clip(N_fail/N_total,0.5,1)`；不进行 per-object reward/friction/grasp-frame tuning。

| 合同项 | Production 值 |
| --- | --- |
| Clip | `--clip {hocap_170105,hocap_170650}` |
| Support | source-first，否则 `INFERRED_PLANAR_SUPPORT` |
| Reward / RSE | `grouped_multiplicative_v1`，RSE enabled |
| RSE scope / kappa 下限 | `0.20 m` / `0.50` |
| Training / evaluation RSI | uniform `[0,320]` / frame 0 full horizon |
| PPO budget | `max_new_updates: 10` upper bound |

qualification 在给定的 `<run_dir>` 下写入 summary、per-episode rows、manifest 和不可变 trace 路径。
replay 仅用于诊断：不会 retrain PPO、改写 trace 或生成新的 qualification。

```bash
conda run -n toporetarget-isaaclab env OMNI_KIT_ACCEPT_EULA=YES \
  python scripts/rl/isaaclab/replay_physical_hoi_trace.py --accept-eula \
  --trace <run_dir>/traces/episode_000.npz --object <clip> --loop
conda run -n toporetarget-isaaclab env OMNI_KIT_ACCEPT_EULA=YES \
  python scripts/rl/isaaclab/replay_physical_hoi_trace.py --accept-eula \
  --trace <run_dir>/traces/episode_000.npz --object <clip> \
  --start-frame <start> --end-frame <end> --no-reference-ghost \
  --mocap-object-low-poly --loop
```

replay 支持 full/windowed trajectory、raw MANO/object overlay、no-reference-ghost 和确定性的
low-poly raw object。详细 reward 语义见 [Grouped reward and RSE](docs/rl/DEXPLORE_STYLE_MULTIPLICATIVE_REWARD_RSE.md)，
support authority 见 [Support resolution](docs/physics/SUPPORT_RESOLUTION.zh-CN.md)。

</details>

### 5. 历史 Raw-Sequence 多 Clip Pilot

本节及链接文档均为 `CURRENT_AUTHORITY=NO` 的历史记录；旧 manifest 不得继续用于 GPU 工作。

旧 raw-sequence batch 入口已删除，因此不能再生成新的 production manifest。历史 receipt 与链接文档
只作为不可变 provenance 保留。请使用上方 EpisodeV1 工作流；不得把历史命令翻译成当前运行。

详见 [独立多 clip 物理精炼](docs/rl/INDEPENDENT_MULTI_CLIP_PHYSICAL_REFINEMENT.md)，其中定义了 authority
manifest、receipt、timing scope 和可执行 promotion 条件。

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

因果物理 PPO pipeline 支持 reference pose、object twist tracking 和版本化 contact reward。
**Aggregate V3 是 legacy additive baseline**（`aggregate_v3`）。**Strict Per-Finger V4 是物理精炼使用的
冻结 contact authority**（`strict_per_finger_v4`），它使用
`SourcePerFingerContactEvidenceV1`：只有 source-confirmed 或 persistent-confirmed 的指定 finger
的 MANO/object contact，才要求同名 Wuji distal/tip body 与 active object 接触。probable、
transition、proximity-only、no-contact 和 ambiguous source state 都不是 V4 的 mandatory contact
semantics。

V4 按 source-required finger 数量归一化独立 named-tip reward。因此其它 finger 的大力不能给缺失的
required finger 记分，source 要求更多 fingers 也不会改变总 contact reward scale。reward 只读取当前
经 filter 的 PhysX named-tip-to-active-object pair force，从不直接控制 object；共享 per-tip force
scale 在 PPO 前由精确的 V1 Formal20 pair-force telemetry 冻结。

物理精炼使用冻结的 **full gravity + active finite support**、nominal friction、object gravity on、
hand/virtual-wrist gravity off，且没有 external object guidance、rollout-time object-state write 或
wrist-root write。PF V2 是 physical functionality authority。当前证据仍仅限单 clip 仿真，
不是硬件或 cross-dataset validation；旧 zero-gravity receipt 保留为历史证据。新精炼配置使用：

```yaml
reward:
  aggregation: grouped_multiplicative_v1
rse:
  enabled: true
```

contact authority 仍须显式指定：

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

## OakInk2 O0–O4 raw-to-physical 准备门

OakInk2 gate 只读取本地 OakInk-v2 program annotation、`anno_preview` 中的
MANO/object track 和 object mesh。它以官方 PrimitiveTask interval 构建 canonical
right-hand record，并冻结 object-disjoint 的 development/certification/held-out split。
它不会进行 retarget、support geometry、physics simulator、frozen-suite evaluation 或 PPO training。

```bash
REPORT_ROOT=.local/"reports"/oakink2_o0_o4_adapter_manifest_v1
conda run -n toporetarget-rl python scripts/data/run_oakink2_o0_o4.py \
  --dataset-root /mnt/nas/storage/Ref2Dex_storage/OakInk2 \
  --report-root "$REPORT_ROOT" \
  --stage all
```

纳入集合被有意限制为：官方 `rh_main` primitive、唯一的官方右手 target、没有左手 object
context、匹配的 object mesh、有效 source track，以及有限的 geometry cross-check。其余 row
都会带 reason 保留在 quarantine manifest 中。该 source/canonical split 独立于 OakInk
可能提供的官方 split。开始 O5 前，必须人工审阅生成的两个 development HTML，并明确批准两者。

### OakInk2 O1R 官方 MANO authority 审计

O1R 在进入 O5 前解析 source-hand authority。OakInk2 `raw_mano` 保存 16 个
scalar-first `WXYZ` quaternion。独立官方路径由 `oakink2_toolkit` 绑定 exact mocap-frame
key，再运行 `ManoLayer(rot_mode="quat", side="right", center_idx=0, use_pca=False,
flat_hand_mean=True)`，并把 `rh__tsl` 加到 vertices 和 21 joints。当前 adapter 在完全相同
的 key 和 licensed MANO asset 上独立运行；两路结果在任何 viewer transform 前做数值比较。

Manifest V1 保持 byte-identical，作为历史证据保留；但其 scalar-last representation metadata
已经使 downstream authority 失效。O1 machine gate 与 corrected O3 geometry cross-check 通过后，
O1R 冻结 Manifest/Split V2，并记录 `SCALAR_FIRST_WXYZ`、MANO asset SHA256、官方 layer
semantics、exact mocap-frame authority 与 O1R authority hash。Machine PASS 不等于 anatomical
validation 完成：同两条 episode 的 Official-vs-Adapter HTML 仍须用户逐条批准。

```bash
conda run -n toporetarget-rl python scripts/data/run_oakink2_o1r.py --help
conda run -n ref2dex-oakink python scripts/data/oakink2_official_reference.py --help
conda run -n toporetarget-rl python scripts/data/run_oakink2_o1r.py --stage all
```

该 CLI 在 `WAITING_FOR_USER_OAKINK2_O1R_HTML_ACCEPTANCE` 停止；不会启动 geometric/Wuji
retarget、support physicalization、PhysX、frozen evaluation 或 PPO。

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
