# Trajectory Continuity Acceptance

Single-frame feasibility and trajectory continuity are separate gates. A
continuous frame is final-accepted only when optimizer convergence, bounds,
active-set feasibility, independent full collision audit, finite values, and
the propagated-state continuity gate all pass.

For `t > 0`, the hard limits are:

| Metric | Limit |
| --- | ---: |
| propagated-relative base translation | 10 mm |
| propagated-relative base rotation | 5 deg |
| propagated-relative finger correction | 0.05 rad |
| excess keypoint displacement | 20 mm |

The artifact records `optimizer_converged`, `single_frame_feasible`,
`trajectory_continuous`, `final_accepted`, failure reasons, initialization
source, retry attempt/profile, and window usage. A solver status of zero and a
collision pass are not sufficient by themselves.
