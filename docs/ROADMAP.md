# Roadmap

This roadmap describes durable research states, not an experiment diary.
Run-specific receipts remain in ignored local storage and detailed stage/RL
documentation.

## Foundation

The stable foundation is canonical HOI data, semantic MANO conversion,
target-hand kinematics, interaction-aware retargeting, geometry/SDF validation,
manifest-bound exports, and browser-based evidence. Arti-MANO and Wuji Hand2
Beta1 remain the supported target-hand lanes.

## Stage16-D causal zero-g milestone (CLOSED)

`CAUSAL_ZERO_G_MILESTONE_COMPLETE`

Stage16-D freezes a simplified Isaac/PhysX reference-tracking baseline:

```text
robot action -> hand-object contact -> object dynamics
```

It is physically causal under this frozen contract: gravity is zero, support is
absent, external object guidance is absent, and rollout-time object-state and
wrist-root writes are forbidden. It is not physically realistic, real-world
calibrated, or full-gravity validation.

### Frozen method state

```text
Aggregate V3
  -> STABLE_BASELINE / global default (aggregate_v3)

Strict Per-Finger V4
  -> EXPERIMENTAL_PARTIAL / explicit opt-in (strict_per_finger_v4)
```

V3 is the stable reference-gated aggregate fingertip pair-force objective.
V4 implements source-side MANO contact semantics and strict independent
per-finger force credit. It improves selected interaction-fidelity,
free-flight, or twist diagnostics, but did not consistently exceed V3 on both
clips' physics qualification; it is therefore not the global default.

### Frozen infrastructure

- Reference Kinematics V2
- PPO-26D Isaac Lab backend
- unified V3/V4 contact-reward configuration
- Source Contact Semantics
- Evaluation Suite V2
- full hand-object pair telemetry
- simulation-data export and replay diagnostics

Historical V1/V2/V3/V4 artifacts remain available through provenance-aware
compatibility mapping. The closeout never reinterprets a historical V4 artifact
as V3 merely because a newer configuration now defaults to V3.

## Next causal physical stage

Create `feature/ppo-physical` only after this milestone PR is merged to
`main`. The required order is:

```text
Contact-ready RSI V2
    ↓
Support Feasibility
    ↓
Gravity + Friction Curriculum
    ↓
Full-gravity / zero-guidance qualification
    ↓
Multi-Clip
```

External guidance or data-H2R remains an assisted fallback after this causal
physics route; it is not a replacement for it.

## Optional assisted data generation

The `feature/guidance-force` experiment adds an explicit bounded
reference-wrench channel for assisted simulation data. It is disabled by
default, outside the causal PPO path, and explicitly an
`ENGINEERING_EXTENSION_ASSISTED_DYNAMICS`; it must not be reported as causal
physics or author-exact TopoRetarget RL.

## Documentation entry points

- [Stage16-D causal zero-g milestone](stages/STAGE16D_CAUSAL_ZERO_G_MILESTONE.md)
- [Stage 16-D physics contract](stages/STAGE16D_PHYSICS_CONSISTENT_RETARGETING.md)
- [Reference Kinematics V2 contract](rl/REFERENCE_KINEMATICS_CONTRACT.md)
- [PPO-26D contract](rl/REFERENCE_TRACKING_PPO_26D.md)
- [Reference-gated contact reward V3](rl/REFERENCE_GATED_CONTACT_REWARD.md)
- [Strict per-finger contact reward V4](rl/STRICT_PER_FINGER_CONTACT_REWARD.md)
- [Source contact semantics](rl/SOURCE_CONTACT_SEMANTICS.md)
- [Evaluation Suite V2](rl/EVALUATION_SUITE_V2.md)
- [Paper fidelity policy](PAPER_FIDELITY.md)
