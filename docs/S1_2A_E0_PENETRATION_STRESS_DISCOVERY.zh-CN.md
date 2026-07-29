# S1.2A E0 穿模压力样本发现

S1.2A 是 paper-external 诊断实验：先发现右手 `artimano_rh` 的 E0 重定向
确实存在机器人碰撞穿模信号的 GRAB clip，再在不改变
`dense_squared_hinge_deadzone1mm_v2`、`lambda=0.1`、1 mm dead zone 的前提下，
比较 E0 baseline 与 S1。

G1 (`s1/airplane_lift`) 只有 source MANO 穿模，E0 robot collision surface
没有有效信号；G2 (`s1/apple_eat_1`) 信号很弱；G3 的 open-mesh/sign 语义和
G4 的 solver/contact 问题仍保持暂停。因此四者在 source eligibility 前排除，
不得由实验结果重新加入。

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src conda run -n topo-retarget python -m toporetarget workflow \
  run-s1-2a-stress-discovery \
  --config configs/experiments/s1_2a_e0_penetration_stress_v1.yaml \
  --experiment-root .local/experiments/s1_2a_e0_penetration_stress_v1 \
  --resume
```

候选扫描、warm probe、E0 probe、冻结锁、完整 E0/S1 结果和 HTML 全部位于
`.local/experiments/s1_2a_e0_penetration_stress_v1/`。fast convex-hull backend
只在 E0 active penetration 区域与 reference triangle-winding backend 对照；
发现漏信号时记录 `FAST_BACKEND_MISS_PENETRATION_SIGNAL`，不修改 loss。

该 stress set 不能代表全 GRAB benchmark，也不能证明全局 default 或 ground-truth
contact retention 改善；自动通过最多只能进入后续受限 lambda study。

该流程与 `docs/SDF_PENETRATION_LOSS.md` 的 S1.2A 小节保持同步：选择只使用
source-only eligibility 与 E0 probe，完整比较固定在相同的 60 帧输入上；不使用
S1 结果回选，不改变 Eq. (8)/(9)、solver 或全局 default。自动通过的状态只允许为
`S1_CONDITIONALLY_ACCEPTED_ON_STRESS_SET`，且仅对冻结 stress set 有效。
