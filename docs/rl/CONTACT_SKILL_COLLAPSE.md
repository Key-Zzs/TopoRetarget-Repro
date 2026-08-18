# Stage16 Contact-Skill Collapse

## Durable conclusion

`ROOT_CAUSE=RESET_DISTRIBUTION_PRIMARY`; `CONFIDENCE=HIGH`.

The frozen V3 `hocap_170105` zero-g actor retains frame-0 approach, contact,
grasp, and lift in the current fixed-wrist C0 runtime when the optimizer is
disabled. Its deterministic 10-episode baseline reaches 10/10 contact, a 0.35
contact fraction, and 10/10 lift success with 0.247 m mean object lift.

The historical failed C0 run stored only its endpoint policy checkpoint. An
exact instrumented reproduction therefore persisted all 26 full
actor/critic/optimizer/normalizer/RNG checkpoints and all 26 exact PPO batches.
Its endpoint is byte-identical to the historical endpoint.

## Localization

The pre-registered contact milestones are:

| milestone | update | stage samples | result |
| --- | ---: | ---: | --- |
| first degradation | 3 | 122,880 | 0/10 contact |
| major collapse | 3 | 122,880 | 0/10 contact |
| first zero contact | 3 | 122,880 | 0/10 contact |
| persistent-zero run starts | 13 | 532,480 | first of three zero rows |
| persistent zero detected | 15 | 614,400 | third consecutive zero row |

U3 is a transient first collapse because U5--U10 recover robust contact and
lift. Robust grasp/lift disappears again at U11 and remains absent thereafter.
At U26, a permissive force threshold observes one grazing frame in each
episode, but contact fraction is only 0.003125, maximum force is about
`1.7e-4 N`, and lift success is zero. This is not robust contact skill and does
not contradict the previous C4 zero-contact qualification.

## Frozen causal A/B

A and B use the same source checkpoint, restored actor, critic, optimizer,
normalizer, RNG, sample counter, C0 physics, reward, controller, reference,
action semantics, sample horizon, and deterministic frame-0 evaluation.
Neither path writes object rollout state or wrist-root state. Only PPO training
reset support differs:

| row | training reset | U3 contact/lift | U6 contact/lift |
| --- | --- | --- | --- |
| A | frame0 only | 0/10, 0/10 | 10/10, 10/10 after transient recovery |
| B | uniform RSI `[0,320]` | 10/10, 10/10 | 10/10, 10/10 |

B remains robust through the pre-registered `A U_ZERO+3` horizon. A already is
a full-state PPO continuation, so a separate continuation row is not a distinct
counterfactual. The near-miss reward row is not required by the frozen decision
tree, and the V3 reward remains unchanged.

## Command and lift interpretation

At the A U2-to-U3 transition, command-to-actual tracking remains small: mean
wrist-position error is 0.514 mm and mean finger error is 0.00987 rad. Contact
loss accompanies distributed policy-command drift and about a 1.1 mm increase
in the unsigned distal-root-to-visual-mesh-vertex distance proxy; it is not
localized to a controller failure. The proxy is not an exact triangle-surface
signed distance and is used only as supporting geometry evidence.

The frozen lift-timing rule also labels actual wrist upward motion before
persistent contact in the source actor. Consequently,
`PREMATURE_LIFT_EMERGES_AT_UPDATE=PREEXISTING_AT_SOURCE`; premature lift is not
an emergent explanation for this collapse.

## Production contract

C0 physical PPO training defaults to uniform RSI over every reference frame
`[0,320]`. Deterministic formal evaluation remains frame-0 full-start. An
explicit `--training-reset frame0` remains available only for reproduction and
ablation. Later physical stages retain their existing frame-0 default.

The bounded fixed-C0 verification stops at U6 / 245,760 samples, before C1. It
preserves 10/10 frame-0 contact and lift at every saved update. This result does
not qualify C1--C4, full gravity, or real-world manipulation.

Run-specific checkpoints, exact PPO batches, traces, formulas, command tables,
and replay commands remain under the ignored local report root:

```text
.local/reports/stage16_contact_skill_collapse/
```
