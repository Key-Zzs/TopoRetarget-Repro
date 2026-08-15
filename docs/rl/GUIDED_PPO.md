# Guided PPO for Assisted Simulation Data

This is an opt-in assisted-data workflow, not causal physical PPO and not an
author-exact TopoRetarget RL result.  The required labels are:

```text
ENGINEERING_EXTENSION_ASSISTED_DYNAMICS
external_guidance=true
assisted_dynamics=true
causal_physics=false
```

The selected `ObjectGuidanceContractV1` injects a bounded, auditable reference
wrench into the dynamic PhysX object. It does not enter the 764D observation,
the 26D policy action, Reward V3, Reward V4, the reference, or the controller.

The guided run must use a copied, SHA-256 checked Reference Kinematics V2
input. `mode=none` is the default and is exactly zero; it preserves legacy
zero-guidance trace loading. `mode=reference_wrench_v1` is only permitted with
V2's corrected timestamps and pose-derived world twists.

Before PPO, run G2 and select one global G3 profile across both clips and
reward modes. Assistance metrics are reporting-only and never modify the
historical V3/V4 checkpoint ranking. Guided exports retain exact contact
telemetry and add wrench/error/clipping/active provenance. The final comparison
must include interaction, twist, penetration, Evaluation Suite V2, and
assistance dominance rather than success rate alone.
