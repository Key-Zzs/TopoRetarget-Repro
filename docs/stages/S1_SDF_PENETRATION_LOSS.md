# Stage S1: dense SDF penetration-loss extension

S1 is a paper-external diagnostic extension over the frozen Stage 9 v3
pipeline. It answers whether a dense signed-distance hinge changes the two
locked GRAB clips under exactly the same source, robot, profiles, constraints,
and full-surface audit.

Required gates:

1. source, object mesh, MANO, robot URDF and collision-surface hashes match the
   locked selection manifest;
2. unit tests pass for lambda-zero construction, monotonic squared hinge,
   geometry-balanced reduction, analytic normal×point-Jacobian gradients, and
   non-finite rejection;
3. the lambda-zero trajectory is numerically equivalent to E0;
4. fixed prescreen frames and the unified lambda are frozen before the full
   comparison;
5. every completed frame has successful strict solver status, feasible active
   constraints, finite values, and a deterministic 512-sample full audit.

The term never reuses paper slack, never adds a clearance margin, and never
substitutes a partial or learned collision surface. A pass is not evidence of
paper-level contact retention; the result remains an external comparison.
