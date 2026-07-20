# Laplacian Interaction Loss

对冻结的 directed source graph，使用 source squared distance 计算 row-normalized weight：

```math
w_{ij}=\frac{\exp(-\kappa\|v_i^s-v_j^s\|_2^2)}
{\sum_{k\in\mathcal N_i}\exp(-\kappa\|v_i^s-v_k^s\|_2^2)},\qquad \kappa=30.
```

Eq. (6) 使用 directed weighted Laplacian：

```math
\Delta(v_i)=v_i-\sum_{j\in\mathcal N_i}w_{ij}v_j,
\qquad r_i=\Delta(v_i^r)-\Delta(v_i^s).
```

Stage 8 的 Eq. (7) 严格为：

```math
E_{IM}=\frac{1}{71}\sum_{i=1}^{71}\|r_i\|_2^2.
```

`InteractionMeshResidual` 用可微 Torch sparse scatter 计算 residual，`InteractionMeshObjective`
提供 loss、scaled residual、per-vertex/hand/object contribution 和最大 residual vertex。evaluation
保存 `[T,213,22]` qpos Jacobian，但不调用 optimizer，qpos/base 与 Stage 7 输入保持不变。

object points 从 source graph 原样复用。identity oracle 的 Eq. (7) 为零，validation 同时检查
`1/sqrt(71)` scaling 和 qpos finite-difference Jacobian。Eq. (8)-(9)、lambda/优化、slack、SDF
与 collision penalty 不在 Stage 8 中隐式执行。
