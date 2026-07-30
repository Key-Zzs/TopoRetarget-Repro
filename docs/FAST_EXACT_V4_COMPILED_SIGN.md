# Fast Exact v4 Compiled Sign

`wuji_continuous_sequential_fast_exact_v4_compiled_sign` is an experimental,
CPU-only, float64, one-thread execution profile.  It preserves the v2 solver,
samples, FD step, active-set policy, signed-distance semantics, generalized
winding definition, strict recovery, and full audit.  It adds the optional
compiled batched winding implementation and certified FD-probe reuse only for
ambiguous spatial FD; normal queries and Stage-12 provenance remain v2
compatible.

v4 is non-default (`recommended: false`, `stage12_default: false`).  It must
pass exactness, deterministic replay, fixed five-frame, and 60-frame gates
before any promotion.  A missing local extension falls back to the established
reference path rather than changing sign semantics.
