# Stage16 Dynamic Physical Qualification

`Stage16DynamicPhysicalQualificationV1` adds an offline acceptance receipt to
the frozen Evaluation Suite V2; it does not modify or reinterpret any
historical V2 receipt.

## Contract

`SR_dynamic` is deliberately distinct from `SR_hold`. This work adds neither a
terminal hold nor a terminal reference freeze, and it makes no PPO, reward,
friction, mass, controller, action, reference, guidance, object-state-write, or
wrist-root-write change.

For every saved control step, it uses the frozen Reference Kinematics V2
world-frame twist convention:

```text
Delta_v     = v_actual - v_reference
Delta_omega = omega_actual - omega_reference
```

The terminal gate reuses the existing frozen V2 terminal-window length and its
contact/free-object numerical limits unchanged. Those limits apply to the
reference-relative residuals above; absolute world-frame terminal zero velocity
is not a new `SR_dynamic` requirement. The velocity reward remains a training
objective, whereas this is an offline acceptance gate.

## Composite result

An episode qualifies only when all of the following are true:

- frozen V2 kinematic success (`SRkin`);
- persistent multi-finger interaction and semantic grasp-and-lift evidence;
- reference-relative terminal twist within the inherited V2 limits;
- frozen absolute-geometry/inter-finger safety, action bounds, and causal
  execution constraints.

The evaluator records the full-motion and legacy-terminal-window mean, p95,
and maximum residuals separately. A terminal-twist failure cannot be hidden by
good full-trajectory averages or by a kinematic pass.

## Evidence boundary

The evaluator consumes only immutable saved traces and their frozen V4
provenance. Its report root is ignored local storage:

```text
.local/reports/stage16_dynamic_physical_qualification_and_grasp_diagnostic/
```

The report retains `SR_dynamic`, historical `SRphysics`, and historical
`SRqualified` as separate fields. It must not be used to promote a PPO lineage,
authorize retraining, or replace Evaluation Suite V2.
