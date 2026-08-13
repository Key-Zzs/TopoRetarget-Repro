# Stage 16-D Reference-Gated Contact Reward V3

## Status and boundary

V3 is the current Stage 16-D causal-reward route. Its research question is
whether rewarding actual fingertip-to-active-object contact when the reference
expects proximity reduces impulse-to-free-flight-to-recatch behaviour while
preserving kinematics and geometry safety.

The causal chain is unchanged:

```text
policy action -> wrist/finger motion -> hand-object contact -> PhysX object dynamics
```

There is no rollout-time object control. `causal_physics=true` means only that
the rollout has no external object control; it does not imply calibrated real-
world physics.

## Contract gates

1. Stage16DReferenceKinematicsV2 must provide 321 factor-eight samples.
2. Both clips must use the same five distal-root landmark and force-column map.
3. The primary reference mask is strict visual unsigned distance `< 0.03 m`.
4. Historical V1 formal inputs must expose exact five fingertip--active-object
   force vectors. Aggregate force/presence is rejected.
5. The pooled exact positive-contact median freezes one shared `lambda_c`;
   fewer than 100 samples blocks PPO.
6. The response is finite, monotonic, bounded by `w_c=1`, and saturating.

V3 has no contact termination. Contact-loss termination, contact-ready RSI V2,
gravity/friction curriculum, and H2R are deliberately outside this stage.

## V1 exact-pair-force calibration

V1 replays both frozen Formal20 sets only to add missing telemetry. At every
control frame it records `[T, R, F, 3] = [321, 20, 5, 3]` pair-force vectors
from the object-side filtered PhysX force matrix, in world-frame N. The five
fingers are thumb/index/middle/ring/pinky, mapped by the runtime asset manifest
to `r_thumb_distal:20`, `r_index_finger_distal:4`,
`r_middle_finger_distal:8`, `r_ring_finger_distal:16`, and
`r_pinky_distal:12`. Each vector is force on the active object from the named
hand collision body.

Frame zero is explicitly invalid rather than treated as a zero-force sample.
Calibration pools only valid frames with an expected finger and positive
`S_contact`, then freezes the cross-clip median once. Aggregate contact force,
contact presence, V2 data, and V3 outcomes are not calibration substitutes.

## Fair protocol

The two clips use their own V1-L0 actor and normalizer, fresh critic and
optimizer, the canonical PPO seed, the same 26-D action/764-D observation,
and unchanged physics/PPO hyperparameters. The four required development
milestones are 1M, 2M, 3M, and 4M samples. A later 8M/12M/16M extension is an
independent per-clip decision based on the frozen effectiveness rule; no run
may exceed 16M.

Formal evaluation consists of 20 unseen deterministic frame-zero episodes per
selected checkpoint. Every episode retains its simulation data even if it is a
diagnostic failure, and only qualified episodes receive qualified labels.

## Replay and data

V3 traces add reference mask, actual mask, exact pair force, force magnitude,
force scale, and `r_contact` while retaining existing V1/V2 replay fields.
They remain replayable by
`scripts/rl/isaaclab/replay_stage16d_simulation_trace.py`; replay is a viewer,
not a physics rollout or qualification.
