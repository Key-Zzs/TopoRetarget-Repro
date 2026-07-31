# Stage-12 static single-frame runtime policy

Stage-12 classifies runtime from the canonical/selection contract, never from a
dataset-name conditional.  A static singleton must declare a static sample
mode, `temporal_metrics_applicable: false`, one articulated frame, and one
canonical frame.

- Dynamic trajectories retain a 90 s per-frame hard gate, a 30 s rolling-p95
  gate, and the consecutive-slow-frame gate.
- Static single-frame samples record a warning from 90 to 300 s and remain
  eligible only if every solver, geometry, sign, and independent-audit gate
  passes.  They hard-stop above 300 s.  Rolling-p95 and consecutive-frame
  gates are `NOT_APPLICABLE`.

The 90 s threshold protects cumulative dynamic trajectory runtime; it is not a
universal terminal gate for a single static sample.  This policy only
classifies health evidence.  It does not alter solver inputs, geometry/SDF,
artifact content, source contracts, or existing dynamic results.

中文说明：静态单帧由 canonical/selection contract 判定，而非数据集名称。动态
轨迹保留 90 秒单帧硬门、30 秒滚动 p95 和连续慢帧门。静态单帧在 90--300 秒
记录警告、仍须通过全部正确性门；超过 300 秒才是硬失败。滚动 p95 与连续慢帧
对静态样本为 `NOT_APPLICABLE`。90 秒门用于防止动态轨迹累计超时，不能作为
静态单帧的统一终止门。该策略不修改求解器、几何/SDF、artifact 或既有动态结果。
