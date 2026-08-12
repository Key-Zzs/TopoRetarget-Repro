# Strict Per-Finger Contact Reward V4

## Scope

`StrictPerFingerContactRewardV4` is the Stage 16-D causal PPO contact contract.
It changes exactly one method variable from the frozen V3 baseline:

```text
V3: reference 3 cm proximity + aggregate named-tip force
V4: source-confirmed per-finger semantics + independent named-tip force
```

The total reward is exactly:

```text
Reward V4 = Reward V2 + r_contact_v4
```

V4 does not inherit or add V3's aggregate contact term. Object pose/link/finger
tracking, wrist tracking, 26-D smoothness, signed object linear/angular twist
terms, action, observation, controller, physics, PPO architecture, and PPO
hyperparameters remain frozen from V2/V3.

## Source mask

For runtime frame `t` and the fixed finger order
`thumb/index/middle/ring/pinky`, the immutable source mask is:

```text
m_src[f,t] = 1 only for SOURCE_CONTACT_CONFIRMED
                      or SOURCE_CONTACT_PERSISTENT
             0 for SOURCE_CONTACT_PROBABLE, SOURCE_CONTACT_TRANSITION,
                   SOURCE_PROXIMITY_ONLY, SOURCE_NO_CONTACT, or ambiguity
```

The source authority is `SourcePerFingerContactEvidenceV1`. Its native source
semantics and factor-eight `41 key -> 321 control frame` mapping are consumed
read-only. The mask is neither regenerated from robot contact nor changed by a
policy result. Each clip stores `strict_source_contact_mask[T,5]`, source
classes, and the fixed finger order.

## Exact force and reward

The named Wuji tips are `r_thumb_distal`, `r_index_finger_distal`,
`r_middle_finger_distal`, `r_ring_finger_distal`, and `r_pinky_distal`. For a
required finger, only its own filtered active-object PhysX pair force is valid:

```text
F[f,t] = || force(tip_f -> active_object)[t] ||_2

r_cf[t] = 0                                      when pair presence is false
                                                   or F[f,t] <= numerical floor
           exp(-lambda_tip / (F[f,t] + epsilon))  otherwise

K[t] = sum_f m_src[f,t]
r_contact_v4[t] = 0                               when K[t] = 0
                  w_c / K[t] * sum_f m_src[f,t] * r_cf[t]  otherwise
```

`w_c=1.0` and `epsilon=1e-5 N` are frozen. The named pair-presence gate and
the formal numerical floor suppress a reward from force-noise samples. Net body
force, whole-hand force, same-finger group sums, aggregate five-finger force,
and contact presence without a valid pair force are not substitutes.

The normalization makes the contact-term upper bound independent of how many
fingers the source requires. More importantly, no thumb/index/ring/pinky force
can reward a missing required middle-finger contact.

## Calibration and information flow

`lambda_tip` is frozen once as the pooled median of positive, named-tip forces
from both clips' V1 Formal20 exact pair-force telemetry, restricted to required
source-mask samples with valid pair presence. Calibration reports per-clip and
per-finger coverage and does not use V3/V4 outcomes. It is shared across
fingers and clips; there is no per-finger or clip-specific scale.

The reward may consume the current actual pair-force/presence and the frozen
source-side reference mask. The actor keeps the 764-D observation and receives
no future actual contact, force, object state, or success signal.

## Causal and evaluation boundary

V4 has no object guidance force/torque, object pose/velocity/angular-velocity
write, attachment, suction, teleport, rollout reset, hidden support, gravity
or friction curriculum, contact-loss termination, terminal reward, penetration
reward, Multi-Clip PPO, or data-H2R.

Development selection uses only the frozen development seeds. Formal20 is an
unseen deterministic frame-zero holdout. In addition to Evaluation Suite V2,
evaluation reports source-tip recall, persistent recall, fully-missing contact,
cross-finger compensation, same-finger non-tip substitution, no-tip/no-hand
flight, recontact, twist, geometry, and per-finger force-farming diagnostics.
