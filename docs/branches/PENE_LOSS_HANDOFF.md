# `develop/pene-loss` handoff

Scope is limited to the dense SDF penetration-loss extension and the bounded
S1 G1/G2 comparison. Changes are intentionally left unstaged, uncommitted and
unpushed for review.

Frozen boundaries: E0 paper weights and constraints, native 120 FPS, fixed
GRAB ranges, right-hand Arti-MANO, full 512 collision samples, strict v3
solver/acceptance policy, no manual acceptance, no G3/G4, no ContactPose, and
no modification of raw inputs or the main worktree.

The handoff must include the selection manifest, input and artifact hashes,
unit-test output, lambda-zero report, fixed prescreen report, unified-profile
decision, per-frame CSV/JSON metrics, full-audit integrity report, and the
self-contained HTML comparison/dashboard.
