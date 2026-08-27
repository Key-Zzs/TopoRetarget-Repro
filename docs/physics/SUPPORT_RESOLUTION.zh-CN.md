# Stage 16 支撑解析与重建

该合同在构建运行时场景之前解析记录序列的物理支撑。它与 PPO、奖励、接触语义以及现有的 hand-object 绝对几何 gate 独立。

## 解析顺序

`auto` 按以下顺序执行：

1. 检查序列目录和源 metadata，寻找显式支撑；
2. 只有 source adapter 提供资产、变换和验证 receipt 时，才验证恢复出的场景几何；
3. 没有经过验证的 source support 时，根据物体 pose/twist、mesh 高度、重力以及可选接触证据，寻找最早的稳定 pre-contact 区间；
4. 拟合重力对齐的平面支撑，同时使用 visual 与 runtime collision mesh 轨迹，并生成有限尺寸的 static/kinematic box proxy；
5. 执行 object/table 与 hand/table 几何检查，再执行匹配的全重力 PhysX A/B：有 proxy 的 object-only 与无 proxy 的 object-only。

有限 footprint 最多可包含稳定区间之后的四帧，但这些帧必须持续满足冻结的
object/table 穿透与间隙限制。首个已经抬起或过度穿透的帧会被记录并排除在
support-contact 几何区间之外；稳定区间本身不会因此缩短。

`source_only` 永远不会回退到推断平面。`inferred_planar` 是显式允许步骤 3–5 的模式。找不到 source asset 不等于证明序列中存在桌面。

## 冻结合同

算法合同位于 [`configs/physics/support_resolution_v1.yaml`](../../configs/physics/support_resolution_v1.yaml)。可复用实现位于 `src/toporetarget/physics/support/`，分别负责 source evidence、平面推断、几何验证、运行时 proxy、PhysX 归约和 fail-closed resolver。

支撑盒体使用局部坐标系生成；Isaac Lab 在 spawn 时通过 `RigidObjectCfg.init_state` 注入已审计的桌面中心与四元数。它具有有限范围和厚度，以及未经标定的 nominal material；不包含力注入、attachment、object teleport、guidance 或 rollout state write。

## 当前 HOCap 资格化结果

对 `hocap_170105` 与 `hocap_170650` 的 source audit 在挂载的序列目录中没有找到可恢复的 source support asset。因此两段都解析为 `INFERRED_PLANAR_SUPPORT`，稳定 pre-contact 区间分别为 `0:18` 与 `0:21`。object/table 几何通过；由于现有 reference 只有 tracked link points 而不是完整 hand collision mesh，完整 hand/table 几何保持 `DEFERRED`。

真实 Isaac Lab 5.1 / GPU PhysX receipt 显示了预期的因果分离：无支撑时两段物体都在全重力下自由下落；有支撑时接触持续存在、法向力约为 `mg`，且位置与四元数姿态漂移都在静态资格化阈值内。因此两段 inferred support 都通过 geometry 与 object-only physics。由于缺少完整 hand collision mesh 且现有 P3 hand-object geometry gate 仍不可用，runtime reference-following transfer 继续延期；这不是通过移动桌面或注入 guidance 隐藏运动的理由。

运行时 reference-following transfer 保持 `DEFERRED_BY_HAND_OBJECT_GEOMETRY`，P3/G3/P4 仍然阻塞。完整 receipt 位于被忽略的 `.local/reports/stage16_support_reconstruction/`。

随后 P3-B.6 将 finite support actor 与完整 21-body Wuji collision reconstruction
一起纳入评估。dynamic reset receipt 中 support contact 与物体稳定性仍然成立，
但 reference trajectory 仍未通过 formal H-O/H-T geometry gate。见
[P3-B.6 物理场景与 RSI 再资格化](../rl/PHYSICAL_SCENE_RSI_REQUALIFICATION.zh-CN.md)；
这不会把 support actor 提升为主 RL environment 的一部分。

## 复现命令

几何推断和可视化：

```bash
PYTHONPATH=src python scripts/physics/prepare_physical_support.py \
  --support auto --static --replay
```

真实 PhysX receipt 的 `with_support` 与 `without_support` 命令见英文合同；对另一个 clip 替换 clip 与资产路径，最后运行：

```bash
PYTHONPATH=src python scripts/physics/summarize_physical_support.py
```

当物理资格化被阻塞时，finalizer 返回非零是预期行为。

## 非目标

本阶段不重新训练 PPO，不修改 reward，不改变 C0/C1/C2/C3/C4 或 G3 gate，不修复 hand-object reference penetration，不向主 RL environment 添加 floor/table，也不把失败的支撑资格化提升为通过。只有 geometry 与 physics 都通过时，支撑 actor 才能从 diagnostic/reconstruction artifact 升级。
