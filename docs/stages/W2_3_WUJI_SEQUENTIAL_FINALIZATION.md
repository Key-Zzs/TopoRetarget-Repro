# W2.3 Wuji Sequential Finalization

W2.3 finalizes a separately named sequential profile for offline reference
generation from the immutable W2.2 Wuji Hand2 Beta1 RH evidence. It does not
rewrite the paper-core profile, formal continuous trajectories, baseline,
canonical source, warm start, interaction graph, historical exports, or the
protected `TopoRetarget-Repro-pene-loss` worktree.

## Profile contract

`wuji_continuous_sequential_v1` is derived from
`wuji_continuous_full_state_v1`. The only solver-semantic change is
`window.fallback_enabled: true -> false`; metadata marks it as a recommended
candidate, an engineering extension, offline-only, `RL_READY=NO`,
`REALTIME_READY=NO`, `CROSS_SUBJECT_VALIDATED=NO`, and
`AUTHOR_EXACT=UNRESOLVED`.

The production sequential path therefore ends after the existing propagated,
trust-region, and deterministic multi-start attempts. The five-frame branch is
only a diagnostic shadow. Its W2.3 harness uses W3 global `[441,446)`, local
`[34,39)`, local frame 34 as a fixed left anchor, normalized coordinates
`0.01 m / 0.1 rad / 0.05 rad / 0.001 m`, analytic block Jacobians, and a
window-local `trust-constr` fallback after scaled SLSQP failure.

## Evidence and gates

The finalization command is:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/deepcybo/miniconda3/bin/python scripts/wuji_w2_3_finalization.py
```

It writes only under
`.local/experiments/wuji_hand2_continuous_v1/w2_3_finalization/`. The evidence
includes input identity and immutability snapshots, structural/profile and
formal-path audits, selected replay frames, an explicit bounded W1 full-replay
status, signed-distance penetration rates at 0/0.25/0.5/1/2 mm, the W3 0.90
to 0.95 explanation, the known-feasible window oracle, deterministic window
shadow evidence, versioned NPZ/Zarr exports, HTML reports, and final integrity
state.

`R_pen(2 mm)` is the hard paper-threshold gate: continuous must not exceed
baseline, maximum depth must be at most 2 mm, and all full-surface and
unqueried audits must pass. The 1 mm rate, p95 depth, and maximum-depth deltas
are secondary warnings. A window failure never blocks the sequential gate.

The output recommendation is scoped to offline reference generation only. A
passing W2.3 result is not evidence of author-exact reproduction, real-time
readiness, RL readiness, or cross-subject validity.
