# Exact SDF and Collision Cache

The strict reference final-refinement profile is immutable. The candidate
`wuji_continuous_sequential_fast_exact_v1` changes execution only: CPU float64,
exact-x callback reuse, cached object geometry, batched collision Jacobians,
and the existing independent full-surface audit. It does not alter Eq. (1)--(9),
collision samples, active QuerySet semantics, bounds, retries, or acceptance.

The candidate remains `recommended: false` and `stage12_default: false` until
function and real-frame parity pass. Its analytic URDF spatial Jacobian has a
reference batched-Torch recovery path; cache keys retain mesh/profile/dtype
identity. A final exact full-surface audit remains mandatory.
