# Batched Collision Jacobians

The existing analytic URDF spatial Jacobian path remains the solver path and
the Torch batched path remains the scalar-reference recovery path. S1.3 does
not alter joint ordering, base-6 columns, floating-point precision, collision
sample identity, or the frozen 512-sample Arti-MANO manifest.
