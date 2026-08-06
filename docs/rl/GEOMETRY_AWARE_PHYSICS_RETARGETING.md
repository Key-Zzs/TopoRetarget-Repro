# Geometry-Aware Physics Retargeting

Stage 16-D.4R2 audits whether the V1 source-relative runtime-collision-proxy
gate is physically attainable before any geometry-aware optimizer or PPO run.
The source and corrected metric implementation is comparable: both use the
same C.1 proxies, transforms, scales, FK, pair filter, python-fcl backend, and
contact-active aggregation. Comparability alone does not validate the
kinematic source overlap as a dynamic-contact bound.

The frozen audit ran a 1,000-repeat backend floor, two 20-replica 321-step
no-contact rollouts, zero-residual dynamic source-following comparisons, and
source-only low-overlap stable-contact trials. Numerical and no-contact floors
passed. The dynamic trials did not maintain required contact topology under
V1, while the stable trials did not establish a 20-replica shared contact
floor. The formal decision is `STAGE16D_GEOMETRY_GATE_REVISION_BLOCKED`.

`RuntimeCollisionProxyPenetrationV2` may only use
`max(source * 1.10 + geometry_epsilon, dynamic_contact_floor)` after a stable
shared floor exists and before optimizer results are viewed. That prerequisite
failed, so V2 was not created. The 10 mm maximum and 3 mm active-p95 absolute
limits are unchanged.

The repository contains fail-closed contracts for online-signal qualification,
exact top-K fallback, hard-gate-first ranking, automatic windows, and bounded
G1/G2 budgets. They are implementation scaffolding, not validated runtime
signals. Exact python-fcl remains the final authority. No geometry-aware
optimizer, demonstration, BC, PPO, two-clip PPO, V2 export, or sensitivity run
is authorized by this closeout.

Factor-8 changes timing semantics. Collision proxies are not visual truth, the
virtual wrist is not a real arm, physical parameters are uncalibrated, and no
real-dynamics or sim-to-real conclusion is supported.
