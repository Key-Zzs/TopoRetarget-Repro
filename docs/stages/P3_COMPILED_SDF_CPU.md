# P3 — Compiled CPU SDF Probe Experiment

P3 implements and qualifies a portable compiled exact closest-point kernel for
the ambiguous 3D spatial-FD probes. It preserves Eq. 1--9, fixed QuerySet
identity, positive-outside signed distance, generalized-winding sign semantics,
and the independent full-surface audit. All generated build and benchmark
artifacts are isolated under `.local/`.

The five-frame result is `COMPILED_KERNEL_LIMITED_VALUE`; P3 is not a Stage-12
precondition and does not alter its queue.
