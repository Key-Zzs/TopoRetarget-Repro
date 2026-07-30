# Compiled SDF CPU Kernel

`compiled_sdf_cpu_v1` is an optional, portable C++17/NumPy-C-API exact
object-local BVH. It is not a paper-specified component and it changes none of
the objective, samples, sign convention, FD step, or final audit.

The handle persists float64 vertices, integer faces, deterministic BVH order,
and query statistics. It performs exact branch-and-bound point-to-triangle
queries with stable face tie-breaking. Inputs must be finite, C-contiguous
`float64` points; invalid inputs raise Python exceptions.

Build it with `python scripts/build_compiled_sdf_cpu.py`. The product is ignored
under `.local/build/compiled_sdf_cpu_v1/`; missing imports safely retain the
v2 Python path.
