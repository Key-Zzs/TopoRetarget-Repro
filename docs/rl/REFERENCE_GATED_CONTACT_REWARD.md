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

## Scale and information flow

`lambda_c` is frozen before training as the pooled positive-contact median of
`S_contact` over the two V1 formal trace sets. At least 100 exact positive
samples are required. A trace that stores only aggregate force and pair
presence is insufficient because it cannot be safely decomposed into the five
pair-force magnitudes; this condition blocks PPO.

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
