# Stage 8 交互图

Stage 8 在 source/robot 边界清晰的前提下实现论文 Eq. (3)-(6)。每帧图在 scene frame `S`
中包含 71 个顶点：0–20 是 canonical MediaPipe-21 hand points，21–70 是固定顺序的 50
个 Stage 6 object face+barycentric samples。

strict profile `configs/retarget/interaction/strict_scipy_qhull_v1.yaml` 对每个 source frame
只调用一次 non-incremental `scipy.spatial.Delaunay`，显式使用 `Qbb Qc Qz Q12`，不使用
`QJ`。centroid translation 和 bounding-box-diagonal scale 只用于 Qhull 数值 conditioning；
source vertex、volume、distance 和 weight 始终保留米制 scene frame。

每个 tetrahedron 的 6 条 edge 完整提取、全局去重并保留。`kappa=30` 从
`configs/paper/retarget.yaml` 读取，directed distance weight 只根据 source vertices 计算并
row-normalize。robot evaluation 复用 source connectivity、weights 和 object points，不运行
robot Delaunay、不重采样、不做 edge filter。duplicate、near-duplicate、coplanar、zero-volume、
isolated 或无 hand-object edge 的 frame 严格失败。

Zarr artifact schema 是 `toporetarget.interaction_graph.v1`，保存 ragged simplex/edge/adjacency、
source Laplacian、hash、profile、frame statistics 和 graph hash。interactive viewer 复用保存的
array，提供 slider、前后帧、play/pause、source/robot、三类 edge、Laplacian/residual/contribution
开关以及 timer cleanup；换帧不重新运行 Delaunay。
