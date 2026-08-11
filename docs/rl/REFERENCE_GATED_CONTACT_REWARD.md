# Reference-Gated Contact Reward

## Scope

`TopoRetargetReferenceTrackingReward26DV3` is the Stage 16-D causal PPO
reward contract. It changes exactly one method variable:

```text
Reward V3 = Reward V2 + r_contact
```

No object force/torque, pose/velocity write, attachment, suction, support,
trajectory servo, contact-loss termination, terminal reward, penetration
reward, curriculum, observation feature, action scale, or PPO architecture
change is part of V3.

## Frozen signal

For the fixed order `thumb`, `index`, `middle`, `ring`, `pinky`, the shared
Evaluation Suite V2 distal-link roots are the five engineering fingertip
landmarks. Their reference-only unsigned distances to the active reference
object's visual mesh define

```text
m_f = 1[D(x_ref,f, O_ref) < 0.03 m]
S_contact = sum_f m_f * ||F_f,active-object||_2
r_contact = 1.0 * exp(-lambda_c / (S_contact + 1e-5))
```

If no `m_f` is active, `r_contact` is explicitly zero. The 2 cm mask is a
diagnostic only; V3 never sweeps or adapts the 3 cm threshold. The visual mesh
is sufficient for unsigned proximity and need not be watertight. A collision
proxy fallback must be labeled as an approximation, not visual contact truth.

`F_f,active-object` is the current PhysX force vector from that fingertip body
to the active manipulated object only. Net fingertip force, self-collision,
palm/other-body force, inactive-object force, support, and scene contacts are
not substitutes. The runtime sensor is an object-side filtered force matrix;
the five selected columns are fixed by name and shared between clips.

## Exact force provenance

The runtime source is `force_matrix_w[N, 1, 21, 3]`: axis 1 is the active
object and the selected hand-collision columns form the world-frame force in
newtons **on that object**. The audited fingertip order is
`thumb/index/middle/ring/pinky`, with the exact link-to-column map
`r_thumb_distal:20`, `r_index_finger_distal:4`,
`r_middle_finger_distal:8`, `r_ring_finger_distal:16`, and
`r_pinky_distal:12`. Names, columns, frame, units, and sign semantics are a
runtime manifest contract, not an unverified positional assumption.

The V1 Formal20 telemetry re-export writes
`replica_fingertip_object_pair_force_world[321, 20, 5, 3]` together with a
`[321, 20]` validity mask. The first frame is invalid because no force sample
has yet been produced; later frames may enter calibration only when valid. The
diagnostic replay preserves the original V1 checkpoint, reference, deterministic
mean action, frame-zero Formal20 seeds, RSI policy, and physics. It adds
telemetry only and never rewrites the historical R7 artifact.

## Scale and information flow

`lambda_c` is frozen before training as the pooled positive-contact median of
`S_contact` over the two V1 formal trace sets. At least 100 exact positive
samples are required. A trace that stores only aggregate force and pair
presence is insufficient because it cannot be safely decomposed into the five
pair-force magnitudes; this condition blocks PPO.

Only a valid frame with at least one expected fingertip and `S_contact > 0`
enters the pool. Both clips must contribute positive samples. The frozen value
is never tuned from V3 outcomes, recalibrated per clip, or replaced by an
aggregate-force estimate.

The reward may read current pair force and current/future reference target.
The actor remains the unchanged 764-D observation and receives no future actual
force/contact, success label, or future actual object state.

## Training and evaluation

Each clip initializes only its own V1-L0 actor and observation normalizer while
resetting critic and optimizer. Development checkpoint selection is independent
of the unseen frame-zero Formal20 set. Evaluation reports standard Evaluation
Suite V2 metrics plus reference-contact recall, persistent-contact recall,
unexpected contact, longest contact-loss gap, recontact count, terminal
contact, and exact fingertip-object force statistics. Reward components are
reported by pre-contact, expected-contact, actual-contact, contact-loss, and
terminal phases.

The pre-training entry point is
`scripts/rl/isaaclab/freeze_stage16d_reward_v3_contact_contract.py`. It writes
the mask/mapping/scale receipts and fails closed before PPO if the pair-force
contract is not fully reconstructible.
