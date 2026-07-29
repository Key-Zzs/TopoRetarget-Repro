# Five-Frame Retargeting Window

When the finite retry cascade cannot produce a continuity-accepted target,
the profile creates a bounded window `[t-1, t, t+1, t+2, t+3]`. The accepted
frame `t-1` is a fixed left anchor, variables are only the in-range future
frames, and only the target frame is committed. Future results are hints and
cannot replace later normal solves. The window is shortened at the clip end,
has one invocation level, keeps per-frame QuerySets/slack/active rounds, and
never expands beyond five frames.

The first implementation uses the same Eq. (8) terms and the same correction
temporal term; it does not add acceleration, jerk, contact, morphology, or
post-filter losses.

## W2.3 diagnostic repair boundary

The W2.3 harness keeps the window experimental-only. It uses the W3 oracle
global `[441,446)`, local `[34,39)`, a fixed left anchor at local 34, normalized
coordinates, analytic block Jacobians, and a window-local `trust-constr`
attempt after scaled SLSQP failure. The current shadow is deterministic but
unresolved (`SLSQP` status 4; the fallback rejects non-finite callback state).
No window result is written into a formal continuous artifact, and window
failure does not block the sequential profile recommendation.
