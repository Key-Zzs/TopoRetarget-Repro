# Source-support feasibility

The physical route distinguishes source-backed support from invented support.
Support is part of the simulation contract, not a convenience stabilization
mechanism.

## Rules

- Use source-backed support only when its provenance and geometry are available.
- Do not replace missing support with a generic floor, table, fixture, or
  attachment.
- Do not inject external guidance or change object/wrist state at rollout time.
- When source support is unavailable, P2 can authorize only the explicitly
  constrained contact-ready route; it cannot authorize a new support model.

This boundary preserves causal attribution: contact behavior must arise from
the hand, the object, and the frozen simulation contract. The P1 safe bank is
evidence that a reset can be initialized under its specified diagnostic, not
evidence that arbitrary support geometry is valid.

The current P3 stop occurs later, at C2 global absolute-geometry selection. It
does not relax the support rule or provide a full-gravity result.
