# Laplacian interaction loss

For a frozen directed source graph, each vertex uses the source-derived
row-normalized weights:

```math
w_{ij} = \frac{\exp(-\kappa\|v_i^s-v_j^s\|_2^2)}
              {\sum_{k\in\mathcal N_i}\exp(-\kappa\|v_i^s-v_k^s\|_2^2)},
\qquad \kappa=30.
```

The implementation evaluates the directed form of Eq. (6):

```math
\Delta(v_i)=v_i-\sum_{j\in\mathcal N_i}w_{ij}v_j,
\qquad r_i=\Delta(v_i^r)-\Delta(v_i^s).
```

The Stage 8 objective is exactly

```math
E_{IM}=\frac{1}{71}\sum_{i=1}^{71}\|r_i\|_2^2.
```

`InteractionMeshResidual` computes the frozen-source residual with a sparse
Torch scatter path that preserves FK autograd. `InteractionMeshObjective`
returns the loss, scaled residual, per-vertex/hand/object contributions, and
the maximum residual vertex. A dense implementation is retained as a numerical
test reference. The saved evaluation artifact includes a `[T,213,22]` qpos
Jacobian, but no optimizer is called and qpos/base are byte-for-byte unchanged
from Stage 7.

Object points are copied exactly from the source graph into the robot vertex
set. The identity oracle therefore has zero Eq. (7) error; validation also
checks the independent `1/sqrt(71)` residual scaling and finite-difference
agreement for bounded qpos Jacobian probes.

This document deliberately stops at Eq. (7). The paper's lambda weighting,
slack variables, signed-distance constraints, collision query set, and final
constrained optimization are not silently folded into this Stage 8 evaluator.
