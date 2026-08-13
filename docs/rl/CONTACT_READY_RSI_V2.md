# Contact-ready RSI V2

Contact-ready RSI V2 is the only reset domain for the physical curriculum. It
uses named safe states produced by the P1 diagnostic rather than arbitrary
reference frames or synthetic pose repair.

## Contract

- Allowed banks are `CONTACT_READY_SAFE`, `PERSISTENT_SAFE`, and
  `MANIPULATION_SAFE`.
- `PRE_CONTACT`, `GRAVITY_RISK`, `INVALID_RESET`, and `AMBIGUOUS` states are
  prohibited.
- The initial physical curriculum must not add a support plane/table, external
  guidance, rollout-time object-state writes, or wrist-root writes.
- Frame-zero full gravity is not an RSI authorization. Full gravity is only a
  later promotion condition.

The loader validates the versioned safe-bank schema and the configured allowed
banks before constructing an environment. Evaluation pairs are also checked so
that every reset belongs to the same safe bank. A missing or incompatible bank
is an error, not a fallback to a nearby reference frame.

## Relationship to support and promotion

P2 support feasibility may authorize constrained contact-ready operation when
source support is absent, but it may not invent support geometry. The staged
curriculum starts from those named banks, evaluates both frozen reward modes at
C2, and requires one global mode to pass both clips before G3. This rule makes
the reset provenance, support boundary, and policy-promotion boundary explicit.

See [Physics curriculum](PHYSICS_CURRICULUM.md) and [support feasibility](../physics/SUPPORT_FEASIBILITY.md).
