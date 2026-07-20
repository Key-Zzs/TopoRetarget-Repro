# Stage 8：交互图与 Laplacian 坐标

状态：`implemented_with_assumptions`，RH/LH bounded 60-frame acceptance 已通过。

## 范围

本阶段只关闭论文 Eq. (3)-(7)：21+50 顶点、source-only Delaunay、完整 tetrahedron edge、
source-derived directed weight、共享 Laplacian、除以 71 的 Eq. (7)、冻结 warm-start 的 qpos
Jacobian/base diagnostic。Eq. (8)-(9)、optimization、slack、SDF、collision、RL 和全数据集评估
仍属于后续阶段。

## 数值假设与边界

strict profile 固定 `Qbb Qc Qz Q12`，不使用随机 jitter。由于 audited sequence 的原始米制
坐标在 Qhull 中出现 zero-volume 数值边界，Qhull 输入使用 deterministic centroid translation
和 bounding-box-diagonal normalization；source vertex、volume、distance、weight 和 artifact
hash 的数据仍在米制 frame。diagnostic jitter profile 不用于 acceptance artifact。

graph builder 不加载 robot 或 warm start；evaluation 加载已保存 graph 和 Stage 7 warm start，
复用 source topology/weights，记录 robot Delaunay=0、optimization=false、SDF/collision=false。
任何 source geometry、object scale 或 sample identity 变化都必须重建 graph。

真实 bounded 产物位于被忽略的 `.local/cache/retarget/` 和 `.local/reports/stage8/`，包括 input
audit、determinism、identity oracle、Jacobian、topology、scale、performance、integrity、validation
和 first/middle/last visualizations。
