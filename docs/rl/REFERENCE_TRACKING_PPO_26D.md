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

## PPO replay trace lifecycle

The deterministic L0 evaluator captures a post-physics row on the GPU for
each control step: physical wrist pose, canonical 20-D finger state, object
state, contact forces/presence, actuator effort, termination reason, action,
reward, and embedded object reference. It does not read collision-body
articulation tensors or make CPU copies from an Isaac callback: that access
can make Isaac Sim 5.1 terminate without a Python exception.

After the rollout, one bulk device-to-host export is made. The evaluator then
uses the frozen Wuji hand FK and the captured physical wrist/finger state to
reconstruct the ordered 21 collision-body poses. It rejects non-finite values
and zero quaternions before writing the trace. The trace records this source
as `offline_fk_from_captured_physical_wrist_and_finger_state`; it is physical
rollout state reconstruction, not an injected reference pose.

## Bounded continuation and qualification

D.5-R5 is a completed 1,024,000-sample `hocap_170650` L0 checkpoint, not a
qualification result. R6A resumes its actor, critic, optimizer, normalizer,
RNG state, and sample count under the unchanged V1 contract to 4M cumulative
samples. The runner refuses a changed fixed clip, 26-D/764-D/PPO contract,
environment count, missing checkpoint state, or a V1 budget above 67,108,864
samples. It writes checkpoint reload receipts, reward components, PPO
requested/actual update counts, KL per epoch/minibatch, and the three action
group diagnostics at every update.

The frozen `development_eval_seed_set_v1` provides 20 deterministic
frame-zero and 20 RSI episodes for `hocap_170650`
L0/4M/16M/curriculum/Reward-V2 comparisons. Its
`formal_holdout_seed_set_v1` is disjoint and prohibited before R7.
`hocap_170105` uses the separate
`development_eval_seed_set_170105_v1` and
`formal_holdout_seed_set_170105_v1` sets; cross-clip seed reuse is forbidden.
The
lexicographic best-checkpoint order is frame-zero task completion, terminal
contact, terminal stability, continuous contact, lower object position/rotation
error, RSI terminal contact, reward, action saturation, then earlier sample
count. Formal R7 is frame-zero only and uses the active runtime collision-proxy
geometry contract; an unqualified trained policy remains a preserved
post-PPO-failure result rather than an authorization failure.

For the completed 170650 ladder, R6B reached 16,793,600 cumulative samples.
The frozen 4M-to-16M gate found only the 14.26% final-object-error improvement;
terminal contact, contact duration, last-contact p75, and RSI terminal contact
did not improve, so the branch stopped at best checkpoint rather than extend to
32M. Development-only selection chose the R6A 2,007,040-sample checkpoint.
Formal R7 classified it as `STAGE16D_170650_PPO_TRAINED_NOT_PHYSICS_QUALIFIED`
(0.70 task success/stability and failed relative geometry comparison). The
subsequent 170105 R8 policy must be fresh and use the unchanged V1 contract.

That fresh 170105 route took the bounded `AMBIGUOUS_ONE_TIME_EXTENSION` at 4M,
then `IMPROVING` at the one permitted 5M extension, and R6B reached 16,793,600
samples. Its 4M-to-16M decision found only the median frame-zero contact-
duration criterion and stopped at the selected best checkpoint. Development
selection chose the 1,024,000-sample L0 checkpoint. Its 20-seed formal R7
result was `STAGE16D_170105_PPO_TRAINED_NOT_PHYSICS_QUALIFIED`: reference
completion and terminal contact were 1.00, but task success and terminal
stability were 0.00, and the relative geometry comparison failed despite
passing the absolute geometry limits. R6C/R6D were not entered; because neither
single-clip R7 physics-qualified, D.6 multi-clip training and D.7 export are
not authorized.
