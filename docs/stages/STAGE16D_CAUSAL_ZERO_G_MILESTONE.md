# Stage16-D causal zero-gravity milestone

## Scope

Stage16-D is closed as a physically causal reference-tracking baseline under a
frozen, simplified Isaac/PhysX contract:

```text
robot action -> hand-object contact -> object dynamics
```

The contract has zero gravity, no support, no external object guidance, no
rollout-time object-state write, and no rollout-time wrist-root write. It is a
causal simulation baseline, not physically realistic, real-world calibrated,
or full-gravity validation.

The tracked, machine-readable contract is
[`stage16d_causal_zero_g_milestone.yaml`](../../configs/rl/stage16/stage16d_causal_zero_g_milestone.yaml).
It records only durable scope and method state, never run paths, checkpoints,
sample counts, or clip-specific results.

## Stable baseline

**Aggregate V3** is `STABLE_BASELINE` and the global default through:

```yaml
reward:
  contact:
    mode: aggregate_v3
```

It is the frozen reference-gated aggregate fingertip pair-force objective and
the baseline for the next physical stage.

## Experimental objective

**Strict Per-Finger V4** is `EXPERIMENTAL_PARTIAL` and requires an explicit
opt-in:

```yaml
reward:
  contact:
    mode: strict_per_finger_v4
```

It uses source-side MANO/object contact semantics and only credits the matching
named fingertip's active-object PhysX pair force. It is fully implemented, but
is not consistently superior to V3 on the two-clip physics qualification and
is not the global default.

## Infrastructure retained

- Reference Kinematics V2
- 26-D PPO residual action and Isaac Lab backend
- unified contact-reward mode/legacy provenance mapping
- Source Contact Semantics and full 21-body pair-force telemetry
- Evaluation Suite V2
- replay diagnostics and simulation-data export

Historical V1/V2/V3/V4 checkpoints, traces, and exported simulation data retain
their recorded provenance. Missing legacy mode fields are mapped only from an
unambiguous V3/V4 reward-contract identifier; there is no hidden fallback.

## Known limitations

- no gravity
- no support
- frozen simplified damping and contact assumptions
- no real-world calibration
- V4 is not consistently superior to V3

## Next

After merge to `main`, the next branch may start the physical curriculum in
this order: Contact-ready RSI V2, Support Feasibility, Gravity + Friction
Curriculum, Full-gravity / zero-guidance qualification, then Multi-Clip.
External guidance/data-H2R remains an assisted fallback after the causal
physics route.

## Fidelity labels

Factor-eight retiming, the 26-D world-wrist extension, V3/V4 objectives, and
the Isaac Lab backend are engineering extensions. They are governed by the
repository's paper-fidelity policy and are not author-exact claims.
