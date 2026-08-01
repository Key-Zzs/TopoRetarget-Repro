# World wrist-and-finger tracking protocol

## Scope and status

`WORLD_WRIST_FINGER_TRACKING_PROTOCOL` is an `ENGINEERING_EXTENSION` to the
preserved Stage-16A `paper_finger_only_base_relative_v1` profile. It exists to
make a world-frame approach trajectory physically actionable in a bounded
MuJoCo experiment. It does not replace the paper/minimal controller, alter
Stage 7–12 artifacts, change raw HO-Cap data, model a real robot arm, or claim
sim-to-real transfer.

The current status is `STAGE16B_BLOCKED_WITH_BOUNDED_EVIDENCE`. The direct
world reference is validated. The finite-wrench wrist controller is partial,
and the clone-only 26D oracle fails the formal wrist-safety gate for both
approved clips. PPO is therefore not started.

## Direct reference contract

The exporter reads accepted Stage-12 `RobotReferenceV2` fields directly:

- `base_pose_scene` -> `T_world_wrist_ref`;
- `object_pose_base` composed with `base_pose_scene` -> `T_world_object_ref`;
- `qpos_reference`, timestamps, source-frame indices, named tracked links,
  and Stage-12 provenance;
- deterministic six object-axis points and the named 16-link Wuji profile.

`WorldWristFingerReferenceV1` resamples all fields to exactly 20 Hz with
linear translation/joint interpolation and quaternion shortest-path SLERP.
Quaternions are active, right-handed `wxyz`, deterministically signed with
non-negative scalar component. The artifact retains both world quantities and
the reconstructed wrist-relative object/axis/link features. Validation fails
if the 20D joint order, provenance/hash, units, cadence, SE(3), or
world-to-wrist reconstruction tolerance is invalid.

The actual generated artifacts are:

```text
.local/stage16_reference_tracking_ppo/world_wrist_references/hocap_170105.world_wrist.stage16.npz
.local/stage16_reference_tracking_ppo/world_wrist_references/hocap_170650.world_wrist.stage16.npz
.local/stage16_reference_tracking_ppo/world_wrist_objects/hocap_170105.obj
.local/stage16_reference_tracking_ppo/world_wrist_objects/hocap_170650.obj
```

## MuJoCo and action contract

The materialized scene adds independent free joints named `stage16_wrist_free`
and `stage16b_object_free`. It uses zero gravity and no synthetic ground. The
object is initialized from the reference only at reset. Formal transitions may
apply an external wrench to `r_wrist`, but never write the object pose or
velocity. `exogenous_wrist_playback_step` and the kinematic-object path are
separately named diagnostics, not formal PPO dynamics.

`a in [-1, 1]^26` has the fixed order:

```text
a[0:3]   wrist translation residual, local frame
a[3:6]   wrist rotation residual, local SO(3) log coordinates
a[6:26]  20 named Wuji finger-joint residuals
```

The selected global scale is 5 mm translation, 2.5 degrees rotation, and 5%
of every finger joint range. The selected global controller has translational
stiffness 250 N/m, damping ratio 1.0, rotational stiffness 15 Nm/rad, damping
ratio 1.0, force limit 25 N, torque limit 1.5 Nm, and feed-forward twist gain
1.0. It is a finite-wrench abstract wrist, not a position teleport.

## Observation, reward, and termination

`WorldWristObservationContractV1` is 764D for 20 joints and 16 tracked links.
It includes wrist tracking error/twist, fingers, previous action, current
world object state, and reference quantities at `[0, 1, 3, 5]` in both world
and wrist-relative frames. It uses one shared running normalizer and no clip
identifier.

The object/base-relative termination profile is preserved. Stage 16-B adds
explicit engineering safety failures at 20 cm wrist-position error and 90
degrees wrist-orientation error. Rewards include original object/link/finger
tracking, wrist position/rotation tracking, and action smoothness. These new
wrist terms and safety limits are assumptions, not paper values.

## Qualification gate

W1 is exogenous wrist playback only. W2 uses a dynamic wrist with a
kinematic-object diagnostic. W3/W4 use the full dynamic free-object system.
The W4 gate uses `WorldWristFingerObjectAwareOracle`, a clone-state only,
finite-difference 26D controller with the same action bounds at H=1, H=5, and
H=10. It has no direct object control.

Each H=10 evaluation runs 20 deterministic frame-0 episodes per clip. PPO
may start only after both clips satisfy the oracle success/final-reach and
object/axis gates. The authoritative result is:

| Clip | H=10 episodes | Success | Final reach | Object position | Wrist position | Wrist rotation | Saturation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `170105` | 20 | 0% | 0% | 0.192 cm | 7.671 cm | 111.055 deg | 100% |
| `170650` | 20 | 0% | 0% | 0.024 cm | 2.247 cm | 123.126 deg | 100% |

Both runs terminate early with `FAILURE_WRIST_ORIENTATION_SAFETY`, before
contact; no trained checkpoint exists.

## Commands

```bash
conda run -n toporetarget-rl python scripts/rl/qualify_stage16_world_wrist.py \
  --reference .local/stage16_reference_tracking_ppo/world_wrist_references/hocap_170105.world_wrist.stage16.npz \
  --reference .local/stage16_reference_tracking_ppo/world_wrist_references/hocap_170650.world_wrist.stage16.npz \
  --object-mesh .local/stage16_reference_tracking_ppo/world_wrist_objects/hocap_170105.obj \
  --object-mesh .local/stage16_reference_tracking_ppo/world_wrist_objects/hocap_170650.obj \
  --scene-root .local/experiments/stage16_world_wrist_finger/qualification_replay \
  --report-root .local/reports/stage16_world_wrist_finger/qualification_replay \
  --formal-episodes 20
```

PPO commands must include the qualifying `--oracle-report`,
`--controller-report`, and `--action-scale-report`; on the current evidence
they fail closed before training. See `train_stage16_world_wrist_ppo.py --help`
for the future-gated CLI rather than inventing a checkpoint path.

## Visualization boundary

`visualize_hocap_world_wrist_policy_mujoco.py` supports zero, oracle, and
PPO policies, interactive or headless modes, an automatic workspace camera,
MP4/PNG/contact-sheet output, and requested reference/contact/wrench display
flags. Existing headless videos show the actual free wrist and object through
the safety failure; they are failure evidence, not visual confirmation of a
successful contact-supported rollout. No single-clip or two-clip PPO video can
exist until the oracle gate passes.
