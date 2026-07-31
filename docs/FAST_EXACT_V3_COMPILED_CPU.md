# Fast Exact v3 Compiled CPU

`wuji_continuous_sequential_fast_exact_v3_compiled_cpu` is experimental,
float64, CPU-only, single-threaded, and non-default. It keeps v2 for normal
queries and uses `compiled_spatial_central_fd_v1` only for ambiguous spatial
central differences. Exact sign classification remains the qualified reference
backend, with graceful v2 fallback if the local extension is absent.

On the frozen five frames, the measured median FD speedup was 1.44x but the
overall median speedup was 1.009x. It is therefore limited value and should
remain experimental rather than replace v2 or a running Stage-12 batch.
