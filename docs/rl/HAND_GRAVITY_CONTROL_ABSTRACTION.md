# Hand gravity control abstraction

## Contract

Stage16 C4 uses nominal world gravity with task objects gravity-enabled. The
Wuji hand and its explicit virtual `3P+3R` wrist are controller-side
articulation links and must be excluded from gravity through the effective
Isaac Lab/PhysX runtime articulation configuration. This is not an object
gravity ablation and is not a per-step state write.

## Why authored USD flags are insufficient

The generated wrapper records per-body gravity-disable opinions for audit
provenance, but imported reduced-coordinate PhysX articulations require a
runtime `RigidBodyPropertiesCfg(disable_gravity=True)` override to make the
contract effective. The override belongs in spawn configuration before scene
construction; physics must never change it in response to contact, phase, or
policy state.

## Boundaries

- Object gravity remains configured independently and is ON during physical
  C4 execution.
- No rollout-time object state or wrist-root state is written by this contract.
- The abstraction does not tune 3R impedance, effort limits, actions, rewards,
  PPO, attachments, or guidance.
- Any future hand-gravity-on experiment is an explicit diagnostic ablation,
  not the production gravity contract.

See [wrist controller root cause](WRIST_ROTATIONAL_CONTROLLER_ROOT_CAUSE.md)
for the decision-tree receipt and regression values.
