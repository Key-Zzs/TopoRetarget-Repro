# Roadmap

This roadmap describes the causal research route, not an experiment diary.
Detailed results and receipts belong in stage/RL documentation and ignored local
machine-readable reports.

## Foundation

The stable foundation is canonical HOI data, semantic MANO conversion,
target-hand kinematics, interaction-aware retargeting, geometry/SDF validation,
manifest-bound exports, and browser-based evidence. The two current target
lanes are tracked Arti-MANO and Wuji Hand2 Beta1.

## Stage 16-D causal physics-consistent retargeting

The causal contract remains:

```text
robot action -> hand-object contact -> object dynamics
```

No object guidance force, hidden controller, object-state correction,
attachment, or suction is part of the main causal lane.

### Phase 1 — Terminal Drift / Support / RSI Attribution (VALIDATED)

Determine terminal-drift provenance with frozen PPO checkpoints and formal
frame-zero episodes. Audit reference terminal twist, actual/residual object
twist, contact impulse/loss, zero-gravity persistence, source/support metadata,
and RSI implementation/state quality. This phase does not train a policy.

### Phase 2 — Evaluation Suite V2 (VALIDATED)

Freeze one additive evaluation contract for single-clip PPO, multi-clip PPO,
future adapters, and physical curricula. It reports `E_r`, `E_t`, `E_j`,
`E_ft`, `SR_kinematic`, `SR_physics`, and `SR_qualified`, while retaining legacy
metrics under their original names. Re-evaluate the existing two frozen
frame-zero baselines with this contract.

### Phase 2.5 — Reference Kinematics V2 (VALIDATED)

Freeze the 41-key spatial specification, timestamps, linear and angular world
twists, and terminal semantics in a single V2 artifact.  The V2 validation
proves that its once-only factor-eight reference resampling preserves physical
time and does not use a runtime re-timing shim.

### Phase 2.6 — V2 Evaluation and Entry Gate (VALIDATED)

Re-evaluate the frozen baselines and Phase 1-R attribution under V2.  This
separates a valid reference target from policy tracking error and authorizes
only the bounded `hocap_170650` Reward V2 experiment.

### Phase 1-R — V2 Attribution Rerun (COMPLETE)

The terminal residual/contact attribution rerun is complete.  It is an
evaluation gate, not a policy-training phase.

### Object Twist Reward V2 (COMPLETED / PARTIAL)

Only after Phases 1–2 establish that reference twist is a valid target and
residual object dynamics are a material terminal failure, version the PPO
tracking reward to add object linear-velocity and angular-velocity tracking.
This phase starts with one causal single-clip retraining/visualization/evaluation
cycle; it does not include contact reward, external guidance, or curricula.

Reference Kinematics V2 and the Phase 1-R attribution passed their entry gates.
The authorized `hocap_170650` Reward V2 P1 probe stopped at its first 1,048,576
sample gate because terminal contact and stability regressed against the frozen
V1 4M baseline. This result does not authorize a 4M/16M continuation or any
reward/physics-contract expansion.

### Reference-Gated Contact Reward V3 (FROZEN BASELINE)

V3 remains the historical aggregate baseline. It uses a reference-only 3 cm
Wuji distal-root proximity mask and an aggregate sum of current named-tip
PhysX pair-force magnitudes. It is preserved for comparison and is not altered
by later source-contact semantics.

### Source Contact Semantics (VALIDATED)

`SourcePerFingerContactEvidenceV1` establishes raw HOCap MANO/object surface
contact per finger and its frozen factor-eight mapping to 321 runtime frames.
It distinguishes confirmed, persistent-confirmed, probable, transition,
proximity-only, no-contact, and ambiguous evidence. The final audit selected
Strict Per-Finger V4 as the unique high-confidence successor to V3.

### Strict Per-Finger Contact Reward V4 (CURRENT)

V4 replaces V3's aggregate contact term while retaining the frozen Reward V2
components, Reference Kinematics V2, 764-D observation, 26-D action, physics,
controller, and PPO hyperparameters. A source-confirmed or
persistent-confirmed requirement for finger `f` can receive contact reward only
from finger `f`'s named distal/tip-to-active-object pair force. The contact
term is normalized by the number of source-required fingers; it cannot use
whole-hand force, same-finger group force, or cross-finger compensation.

The V4 causal route is:

```text
Source Contact Semantics
    VALIDATED
        ↓
Strict Per-Finger Reward V4
    CURRENT
        ↓
if validated:
    Freeze Causal Contact Reward
        ↓
    Contact-ready RSI V2
        ↓
    Support Feasibility
        ↓
    Gravity + Friction Curriculum
        ↓
    Full-gravity / zero-guidance Formal Qualification
        ↓
    Multi-Clip
        ↓
    causal milestone
        ↓
only if causal path insufficient:
    external guidance / data-H2R
```

No V4 phase adds object guidance, object-state writes, attachment, suction,
contact-loss termination, terminal reward, penetration reward, gravity or
friction curriculum, Multi-Clip PPO, or data-H2R. Gravity and physics
curricula are deliberately before any external-guidance route.

## After the milestone

### `feature/ppo-adapter` (FUTURE)

For each existing geometry adapter, choose two representative data sequences,
run PPO, create visualization, and evaluate with Evaluation Suite V2.

### `feature/ppo-physical` (FUTURE)

Advance physical realism with contact-ready RSI V2, support reconstruction,
gravity/friction curricula, mass/inertia uncertainty, dynamics randomization,
and sensitivity analysis.

## Documentation entry points

- [Stage 16-D physics contract](stages/STAGE16D_PHYSICS_CONSISTENT_RETARGETING.md)
- [Terminal dynamics attribution](stages/STAGE16D_PHASE1_TERMINAL_DYNAMICS.md)
- [PPO-26D contract](rl/REFERENCE_TRACKING_PPO_26D.md)
- [Evaluation Suite V2](rl/EVALUATION_SUITE_V2.md)
- [Reference Kinematics V2 contract](rl/REFERENCE_KINEMATICS_CONTRACT.md)
- [Phase 3 object-dynamics reward](stages/STAGE16D_PHASE3_OBJECT_DYNAMICS_REWARD.md)
- [Reference-gated contact reward V3](stages/STAGE16D_CONTACT_REWARD_V3.md)
- [Strict per-finger contact reward V4](rl/STRICT_PER_FINGER_CONTACT_REWARD.md)
- [Stage 16-D Strict Per-Finger V4](stages/STAGE16D_STRICT_PER_FINGER_V4.md)
