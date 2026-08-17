# Stage16 wrist rotational controller root cause

## Status

`PHYSX_ARTICULATION_GRAVITY_OVERRIDE_MISSING` is fixed on
`feature/ppo-physical`. This is an engineering controller repair, not a
post-repair PPO or scientific-qualification result.

## Root cause and minimal repair

The explicit virtual wrist is an imported reduced-coordinate `3P+3R`
articulation. Its generated USD contains per-body `disableGravity` opinions,
but a live PhysX reproduction established that those authored opinions were
not effective for the imported articulation under C4 world gravity. The three
rotational joints then reached the 500 Nm effort limit and failed to track.

`configure_explicit_virtual_wrist()` now applies
`RigidBodyPropertiesCfg(disable_gravity=True)` to the robot spawn at Isaac Lab
runtime. This is an articulation-wide hand/wrist gravity exclusion only;
objects retain the separately configured C4 gravity-on contract. No gains,
action range, reward, PPO state, object rollout state, wrist-root state, or
guidance path changed.

## Decision-tree evidence

- Asset-derived `R_cmd -> FK(q_target)` reconstruction and C4 target
  feasibility pass; rotation targets are radians and stay inside `[-179, 179]`
  degrees.
- All `+/-5`, `+/-15`, and `+/-30` degree single-axis 3R probes respond in
  free space. A C4 frame-zero mixed rotation plus real finger target also
  passes when gravity is effectively disabled.
- In the production C4 scene, disabling self collision, isolating objects, or
  removing table actors does not remove the large rotational drift; collision,
  object, and table are therefore not primary causes.
- The exact standalone articulation fails in world gravity with USD opinions
  alone (55.76 degree mixed-target joint error and 500 Nm saturation), and
  passes with the runtime override (0.11 degree error without saturation).

## Regression boundary

The fixed C4 static hold reaches 0.028 degree mean wrist error rather than
about 70.6 degrees. PPO-off reference following has command-to-actual rotation
means of 0.408 degrees for `hocap_170105` and 0.292 degrees for
`hocap_170650`. The four immutable C4 actors replay for 321 finite frames with
optimizer step zero. Those old-policy traces only verify the controller repair;
they do not requalify reward, contact, object retention, or scientific claims.

Machine-local receipts and GUI/headless replay commands are under
`.local/reports/stage16_wrist_rotational_controller_repair/`.
