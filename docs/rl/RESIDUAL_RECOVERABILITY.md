# Residual recoverability and physical qualification

The Stage 16 geometric retarget reference is immutable and may contain a
physical defect.  PPO observes it as a soft target through the frozen 26-D
residual mapping: local wrist SE(3) residual plus 20 bounded finger residuals.
It is not valid to write a feasibility solution back into that reference.

The boundaries are deliberately separate:

- Reference geometry is a diagnostic used to identify where a residual must
  have authority.
- A reset state must pass exact hand-object, hand-table, object-table,
  inter-finger, joint-limit, and 1g dynamic checks.
- The actual policy rollout must pass the same physical geometry and causality
  gates.

Any controller-limit projection is confined to the final actuator command
boundary.  It cannot move the hand out of collision or otherwise become a
geometry-aware rollout override.
