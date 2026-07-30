# Compiled Generalized Winding

`compiled_batched_generalized_winding_v1` is an optional portable C++17,
single-threaded, float64 implementation of the existing triangle solid-angle
generalized winding definition.  The handle owns an immutable object-local
triangle array and reduces triangles in deterministic input order, without a
Python point loop or an N-by-triangle temporary allocation.

It preserves the positive-outside signed-distance convention and the existing
winding threshold.  Values in the configured confidence band around the
threshold, and all non-finite values, are classified by the qualified Python
reference backend.  It is not a ray-parity replacement and is not paper
specified.

The optional local build is produced by `scripts/build_compiled_sdf_cpu.py`
under `.local/build/compiled_exact_sign_v1/`; import or build failure retains
the v2/v3 reference path.
