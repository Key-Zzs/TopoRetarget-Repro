# Stage 9.2 — refinement performance and recoverability

Status is determined from real evidence, not from the presence of optimization
code. The Stage 9.1 blocked baseline remains preserved in
`.local/reports/stage9_performance/stage9_1_blocked_baseline.json` and is not
overwritten.

The implementation boundary is:

```text
immutable frame context
  -> exact-x cache and callback accounting
  -> persistent mesh/reference/solver SDF resources
  -> batched collision FK/Jacobian
  -> scheduled independent 512-point audits
  -> atomic accepted-frame checkpoints
  -> contiguous resume and final assembly
  -> fresh/resumed/repeat array comparison
```

The first frame still uses the Stage 7 seed. The v2 active-set continuation
still uses the previous `result.x` and query-ID slack remapping. No neighboring
frame initialization is silently promoted to the default. All optimized
paths must preserve the reference objective/gradient/constraint/Jacobian and
strict acceptance within the tolerances defined by the Stage 9.2 objective.

Required evidence under `.local/reports/stage9_performance/` includes
`execution_path_audit.json`, `benchmark_frames.json`, per-frame profile JSON,
callback/cache reports, cProfile and bounded torch-profiler outputs, checkpoint
status, deterministic comparison, and the final performance-gate report. A missing
`py-spy` executable is recorded as unavailable and is not a blocker.

The contact-rich 60-frame run is allowed only after the fixed benchmark set and
checkpoint interruption/resume tests pass the projected runtime gate. A run
that exceeds the gate, fails strict acceptance, fails numerical equivalence, or
cannot complete 60 frames is reported as
`STAGE9_2_IMPLEMENTED_RUNTIME_GATE_BLOCKED`. The completed run instead meets
the reference-runtime minimum gate and is recorded as
`STAGE9_2_COMPLETE_REFERENCE_RUNTIME`; Stage 10 remains blocked because the
preferred single-frame gate is not met and this stage does not authorize it.

The v3 contact-rich run uses execution profile
`cached_checkpoint_cpu_float64_v3` (analytic URDF spatial Jacobian, strict
reference recovery, and SDF tree leaf size 512). The first 60-frame artifact
measures median `10.766 s`, p95 `38.711 s`, and `1104.827 s` total solve time;
the deterministic repeat measures median `10.773 s`, p95 `39.052 s`, and
`1107.368 s`. Both runs have 60/60 strict-accepted status-0 frames, valid
checkpoint chains, and independent 512-sample reference validation with maximum
signed-distance error `2.50e-16 m`. The full persisted arrays are exactly equal
between runs after excluding `solve_time_s` and documented metadata. The recorded
status is therefore `STAGE9_2_COMPLETE_REFERENCE_RUNTIME`; the preferred
single-frame gate is still unmet and Stage 10 remains blocked.
