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
