# Stage 16-D.5 Reference-Residual PPO-26D

## Status and scope

`TOPORETARGET_PPO_REPRODUCTION_WITH_26D_WRIST_ADAPTATION` is paper-guided
reference-tracking PPO, not exact author reproduction. It uses current
Stage16-D engineering physics: a factor-8 321-sample 20 Hz reference, 120 Hz
PhysX with decimation 6, a free object, self-collision, the explicit serial
3P+3R virtual wrist, and 20 finger joints. PPO does not change mass, inertia,
friction, collision geometry, controller gains, effort limits, timing, or
terminal definitions.

The former S3/CEM `0/20` terminal/contact/success results are
`PRE_PPO_BASELINE_FAILURE`. They are Gate C diagnostics, not PPO entry
conditions.

## Action contract

`Stage16DReferenceResidualAction26DV1` has 26 policy coordinates:

| Slice | Semantics |
| --- | --- |
| `action[0:3]` | wrist translation residual in reference-wrist local frame |
| `action[3:6]` | wrist rotation-vector residual (exponential coordinates) |
| `action[6:26]` | canonical 20-D finger joint-position residual |

At reference index `k_t`, the wrist target is
`T_wrist_reference[k_t] ⊕ DeltaT(action[0:6])`, then the existing SE(3)-to-
explicit-3P+3R adapter produces virtual-wrist targets. The finger target is
`q_finger_reference[k_t] + DeltaQ(action[6:26])`, clamped by existing limits.
Policy actions never directly command virtual-wrist articulation coordinates.
There is no object action, object/wrist rollout state write, attachment,
suction, or hidden force.

## Observation, RSI, reward, and gates

`Stage16DPPO26DObservationV2` freezes a 764-D policy vector containing
current/reference/error wrist pose and current/reference wrist twist, fingers
and previous 26-D action, current object state, and current/+1/+3/+5 reference
object/links/fingers. It excludes future actual state, CEM candidates,
penetration future, and success labels.

`Stage16DPPO26DRSIV1` samples valid reset indices uniformly, writes wrist,
fingers, and the free object only during reset, and advances exactly one
reference sample per control step. `TopoRetargetReferenceTrackingReward26DV1`
uses paper object-axis, tracked-link, finger-joint, and whole-26D smoothness
terms, plus 2 cm wrist position and 10 degree wrist rotation tracking. It has
no terminal-contact, penetration, special-clip, or semantic task bonus.

Gate A validates trainability and is the only PPO authorization. Gate B
monitors safety during training. Gate C performs post-PPO
task/contact/geometry qualification.
