# Wuji Five-Frame Window Harness Repair

The W2.2 window fallback is kept as an experimental branch. W2.3 does not
replace any formal artifact with a window result.

## Oracle contract

The known-feasible W3 oracle is global `[441, 446)`, local `[34, 39)`. Local
frame 34 is the fixed left anchor; variable frames are 35, 36, 37, and 38.
The indexing audit covers q/base/slack offsets, QuerySet offsets, object pose,
warm state, chart state, and accepted-frame concatenation. All oracle checks
pass.

## Repair changes

The diagnostic harness adds a fixed-left-anchor temporal term, normalized
coordinates (`0.01 m / 0.1 rad / 0.05 rad / 0.001 m`), and an analytic block
Jacobian. A window-local `trust-constr` attempt is made only after scaled SLSQP
fails. Future states remain hints; only the center is diagnostically compared.

## Current result

Two deterministic shadow runs reproduce the same unresolved result:

| Solver | Status | Feasible | Center continuity | Full audit | Deterministic |
| --- | --- | --- | --- | --- | --- |
| scaled SLSQP | status 4, inequality constraints incompatible | no accepted repair | diagnostic formal center passes | formal artifact untouched | yes |
| trust-constr fallback | rejected with `ValueError: array must not contain infs or NaNs` | unresolved | nonblocking | formal artifact untouched | yes |

The machine status is `WINDOW_FALLBACK_EXPERIMENTAL_UNRESOLVED_NONBLOCKING`.
This failure does not block the sequential recommendation. Evidence is in
`w2_3_finalization/window_oracle/` and
`w2_3_finalization/window_experimental/`.
