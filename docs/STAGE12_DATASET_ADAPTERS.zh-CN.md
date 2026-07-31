# Stage 12 数据集适配器

Stage 12 将四个已解压数据集统一经过同一条冻结契约：

```text
原始数据集 -> CanonicalHOI v2 -> MediaPipe21 -> wuji_hand2_beta1_rh
                                      -> wuji_continuous_sequential_v1
```

适配器按轨迹惰性读取。`discover`、`index`、`describe` 只读 manifest 或元数据；
`load_sequence` 只读取选定帧、选定 MANO/物体标注及必要网格。不会下载、重新解压、
复制完整数据集、重采样时间或加入数据集专用重定向权重。

Stage 12 集成工作树为 `integration/dataset-adapter-v1`；对应的本地 adapter 分支为
`feature/dataset-dexycb`、`feature/dataset-oakink`、`feature/dataset-hocap` 和
`feature/dataset-contactpose`。共享集成仅包含 adapter 注册、冻结选择执行器、测试和本文档；
不改变 robot、solver、viewer 或 metrics core 的行为。

| 数据集 | 适配器 | 原始契约 | 选定轨迹 |
| --- | --- | --- | --- |
| DexYCB | `DexYCBAdapterV1` | `labels_*.npz`、`meta.yml`、YCB 网格 | pitcher、power drill |
| OakInk | `OakInkAdapterV1` | `seq_all.json`、`hand_v`、`obj_transf`、物体网格 | 两条 A01001 序列 |
| HO-Cap | `HOCapAdapterV1` | `poses_m.npy`、`poses_o.npy`、`meta.yaml`、物体部件 | G10、G04 |
| ContactPose | `ContactPoseAdapterV1` | annotation JSON、MANO fit JSON、物体 PLY | mug、banana |

冻结选择和半开帧区间见
[`configs/benchmarks/stage12_selection.yaml`](../configs/benchmarks/stage12_selection.yaml)。
每条数据集 manifest 包含源路径、版本、索引哈希、license 状态、序列数和能力信息。

## Canonical 与溯源规则

所有源 MANO 几何都使用共享 MANO 模型渲染，再通过已有的语义
`mano_v1_2_smplx_to_mediapipe21` 转换器得到 MediaPipe21；禁止仅凭数组形状重排关节。
源坐标系、源哈希、选定帧区间、物体 ID 和转换约定均保存在 `CanonicalHOI v2` 溯源中。

当前 ContactPose 没有经过核验的官方手部骨骼归属关系。因此元数据固定为
`contact_annotation_available=false` 和 `contact_benchmark_status=NOT_AVAILABLE`，
适配器不会伪造 Eq. 10/11 接触分数。

## 产物与报告

所有产物隔离在
`.local/experiments/stage12_dataset_validation/<dataset>/<sequence>/`，包括
`canonical/canonical_hoi_v2.zarr`、`warm/warm_start.zarr`、
`final/final_retarget.zarr`、物体采样、interaction graph、Wuji 碰撞采样、HTML 和
JSON/Markdown 报告。
报告包含 Ebone、EIM/RMSE、各手指 RMSE、穿透、连续性、运行时间和求解/审计计数，并给出
失败分类。HO-Cap canonical 保留全部物体部件。多部件 selection 必须显式给出唯一的
`primary_object`；adapter 将其标记为 `primary_manipulation_object`，共享 Stage 8/9 路径
不再从数组顺序猜测目标。其余声明部件是 context geometry，并持续显示在 source
qualification HTML 中。

ContactPose 是物体坐标中的单帧静态 MANO fit，source 变换固定为 `inv(mTc)`。
RGB-D annotation 中 moving-hand 的 `hTo` 仅保留为刚体观测证据，绝不再复合进静态 MANO
顶点或关节。

```bash
PYTHONNOUSERSITE=1 \
  /home/deepcybo/miniconda3/envs/topo-retarget/bin/python \
  scripts/stage12_dataset_validation.py
```

可用 `--dataset`、`--selection-index` 或 `--max-trajectories` 做有界重跑。脚本只写入
`.local`，不提交生成产物。适配器真实 NAS smoke test 需要设置
`STAGE12_RUN_NAS_TESTS=1`，并标记为 `licensed_data`。所有有界运行完成后，不带筛选
参数执行 `--aggregate-only` 可在不重跑求解的前提下重建八条轨迹的
`stage12_summary.json` handoff，其中包含 Wuji 完成率、报告和 HTML 路径以及逐轨迹指标。
