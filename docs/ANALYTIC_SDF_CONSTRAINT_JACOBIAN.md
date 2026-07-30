# Analytic Signed-Distance Constraint Jacobian

`wuji_continuous_sequential_fast_exact_v2` keeps the existing constrained objective and
uses `J_phi = grad_x(phi)^T J_point` for each active collision sample. At a stable
closest feature, `grad_x(phi) = sign(phi) * (x - closest_point) / unsigned_distance`.
This is positive-outside in both object and scene coordinates; it is not a face-normal
substitution. Hard slack columns are zero and the matched soft-slack column is one.

Near-surface, edge/vertex, sign-unreliable, non-finite, and non-smooth rows are routed
only to fixed-step three-dimensional central differences. They never use optimizer-
coordinate finite differences in v2. A non-finite FD result is explicit diagnostic
`SURFACE_NORMAL_LAST_RESORT` and makes qualification fail closed.
