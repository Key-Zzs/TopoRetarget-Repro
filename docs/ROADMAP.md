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

### Phase 1 — Terminal Drift / Support / RSI Attribution (CURRENT)

Determine terminal-drift provenance with frozen PPO checkpoints and formal
frame-zero episodes. Audit reference terminal twist, actual/residual object
twist, contact impulse/loss, zero-gravity persistence, source/support metadata,
and RSI implementation/state quality. This phase does not train a policy.

### Phase 2 — Evaluation Suite V2 (CURRENT)

Freeze one additive evaluation contract for single-clip PPO, multi-clip PPO,
future adapters, and physical curricula. It reports `E_r`, `E_t`, `E_j`,
`E_ft`, `SR_kinematic`, `SR_physics`, and `SR_qualified`, while retaining legacy
metrics under their original names. Re-evaluate the existing two frozen
frame-zero baselines with this contract.

### Phase 3 — Object Twist Reward (NEXT IF AUTHORIZED)

Only after Phases 1–2 establish that reference twist is a valid target and
residual object dynamics are a material terminal failure, version the PPO
tracking reward to add object linear-velocity and angular-velocity tracking.
This phase starts with one causal single-clip retraining/visualization/evaluation
cycle; it does not include contact reward, external guidance, or curricula.

### Phase 4 — Causal Decision Tree (FUTURE)

| Observation | Next causal correction |
| --- | --- |
| Contact repeatedly breaks | Reference-gated contact reward plus hysteretic contact-loss termination |
| Contact is sound but terminal dynamics are unstable | Contact-ready RSI plus gravity/friction curriculum |
| Causal performance is acceptable | Freeze the global causal contract, run the second single clip, then Multi-Clip PPO and a milestone PR to `main` |
| Bounded causal corrections remain insufficient | Create a separate future `develop/data-H2R` assisted-data branch |

The assisted branch is not the main causal solution. Its results must identify
external guidance and declare `assisted=true`, `causal_physics=false`.

## Milestone

```text
causal single clip -> causal second clip -> Multi-Clip PPO -> milestone PR to main
```

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
