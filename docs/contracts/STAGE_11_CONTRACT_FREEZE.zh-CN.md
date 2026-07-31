# Stage 11 契约冻结

Stage 11 冻结未来 dataset adapter、robot-hand plugin、playback 和
reference-tracking 使用的接口。本阶段不增加数据集、不运行重定向、不修改 solver、
也不修改论文公式。

## 契约结构

| 契约 | Python 入口 | 序列化版本 | 当前实例 |
| --- | --- | --- | --- |
| Canonical HOI | `toporetarget.contracts.canonical` | `toporetarget.hoi.v2` | GRAB 从 v1 迁移 |
| Dataset adapter | `toporetarget.contracts.dataset` | `toporetarget.dataset_adapter.v1` | GRAB |
| Robot hand plugin | `toporetarget.contracts.robot` | `toporetarget.robot_hand_plugin.v1` | Arti-MANO、Wuji Hand2 Beta1 |
| Robot reference | `toporetarget.contracts.reference` | `toporetarget.robot_reference.v2` | NPZ、Zarr |
| Metric registry | `toporetarget.contracts.metrics` | `toporetarget.metric_registry.v1` | paper exact、proxy、geometry、diagnostic |

未来代码必须通过对应 registry 注册 dataset 或 robot。不能在 canonical schema、
retarget solver 或 reference exporter 中增加 dataset/robot 名称判断。Dataset proxy
必须声明为 `DATASET_PROXY`，不能伪装成 ground truth。

## 兼容性

`migrate_v1_to_v2()` 读取已有 `toporetarget.hoi.v1` cache，并返回 v2 facade，
不会修改 source。`load_canonical_hoi()` 同时支持未标记的历史 v1 cache 和 v2 marker。
已有的 `toporetarget.data.*`、`toporetarget.robots.*` 以及 metric import 继续有效。

`RobotReferenceV2` 保存 `qpos_reference`、scene-frame base pose、robot-base 中的 object
pose、robot-base 中的 tracked link positions、timestamps/FPS、显式 joint order、robot
hash 和 dataset provenance，并支持 NPZ 与仓库兼容的 Zarr。

## 当前边界

当前 Wuji 状态是 offline reference generation ready；不是 RL ready、不是 realtime、
也没有 cross-dataset validation。这些声明属于后续阶段，必须分别通过证据 gate。
