# World wrist-and-finger tracking protocol

## Scope and status

`WORLD_WRIST_FINGER_TRACKING_PROTOCOL` is an `ENGINEERING_EXTENSION` to the
preserved Stage-16A `paper_finger_only_base_relative_v1` profile. It exists to
make a world-frame approach trajectory physically actionable in a bounded
MuJoCo experiment. It does not replace the paper/minimal controller, alter
Stage 7–12 artifacts, change raw HO-Cap data, model a real robot arm, or claim
sim-to-real transfer.

The MuJoCo lane is closed with
`STAGE16B_ADAPTIVE_MULTI_HORIZON_ORACLE_PARTIAL`. The direct world reference,
finite-wrench W2 wrist controller, and per-trajectory fixed-horizon evidence
are complete. A shared state-adaptive H1/H5/H10 oracle passes `170650` but not
`170105`; the one global 48x4 upgrade does not change that decision. PPO is
`NOT_STARTED_GATE_BLOCKED`, never `PPO_TRAINING_FAILED`.

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

The selected global scale is 10 mm translation, 5 degrees rotation, and 10%
of every finger joint range. The selected global controller has translational
stiffness 250 N/m, damping ratio 1.0, rotational stiffness 2 Nm/rad, damping
ratio 0.5, force limit 25 N, torque limit 1.5 Nm, and feed-forward twist gain
1.0. The lower rotational stiffness keeps the 10 ms physics integration
stable for the measured free-joint effective inertia. It is a finite-wrench
abstract wrist, not a position teleport.

Reference twists use world-frame linear and angular velocity. MuJoCo
free-joint `qvel` uses world-frame linear velocity but body-local angular
velocity. Reset, observation, playback, object-diagnostic, and impedance-
controller boundaries convert explicitly between those representations.

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

## Object dynamics and support audit

Both derived OBJ meshes are non-watertight, and the frozen source artifacts
provide no authoritative object mass, inertia, support, or force metadata.
The 50 g MuJoCo mass is therefore retained only as a globally shared
engineering assumption; bounded 0.5x/1x/2x/5x mass-inertia counterfactuals are
reported but are ineligible for score-based selection. The formal scene has
zero gravity, no ground, no support constraint, and zero object free-joint
damping. Its classification is
`UNSUPPORTED_FREE_BODY_ZERO_GRAVITY_NO_DAMPING` and physical provenance
remains `OBJECT_DYNAMICS_PHYSICAL_PROVENANCE_UNRESOLVED`.

Contact is audited at every 10 ms physics substep, including hand-object
normal force, impulse, penetration, and geom pairs. This matters because the
`170650` zero-residual trajectory has a two-substep impact that is absent at
the 50 ms control-frame endpoint. In the unsupported model, such a brief
off-centre impulse creates object twist that persists after contact is lost.
The geometric reference has non-zero inertial-wrench demand on 37/41 and
40/41 frames respectively, but contains no force/support provenance. This is
the supported root-cause statement; it is not a proof that the source motion
is physically infeasible.

## Qualification gate

W1 is exogenous wrist playback only. W2 uses a dynamic wrist with a
kinematic-object diagnostic. W3/W4 use the full dynamic free-object system.
The W4 gate uses `WorldWristFingerObjectAwareOracle`, a clone-state-only,
deterministic contact-aware CEM MPC over a true H-by-26 action sequence at
H=1, H=5, and H=10. The selected shared bounded budget is population 32,
three iterations, and eight elites. Receding-horizon execution applies only
the first 26D action through `env.step`; it has no direct object control.

Each H=10 evaluation runs 20 deterministic frame-0 episodes per clip. One
episode computes the MPC trace and 19 episodes replay that exact trace only
through `env.step`; full MuJoCo integration state is restored for every clone,
and all 20 terminal tuples are identical. PPO
may start only after both clips satisfy the oracle success/final-reach and
object/axis gates. W2 passes on both clips with the shared controller:

| Clip | Success | Final reach | Wrist position | Wrist rotation | Force saturation | Torque saturation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `170105` | 100% | 100% | 0.757 cm | 0.417 deg | 0% | 0% |
| `170650` | 100% | 100% | 0.691 cm | 0.906 deg | 0% | 0% |

The frozen shared adaptive-oracle result is:

| Clip | Episodes | Success | Progress | Object position | Object rotation | Max axis | Wrist rotation | Saturation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `170105` selected 32x3 | 20 | 0% | 80.0% | 3.406 cm | 22.497 deg | 5.194 cm | 3.165 deg | bounded |
| `170650` selected 32x3 | 20 | 100% | 100% | 0.119 cm | 0.667 deg | 0.158 cm | 0.877 deg | bounded |
| `170105` bounded 48x4 upgrade | 20 | 0% | 82.5% | 3.637 cm | 18.192 deg | 5.002 cm | 4.205 deg | bounded |

`170105` deterministically terminates with `FAILURE_OBJECT_AXIS_POINT`;
`170650` passes. The unique global 48x4 upgrade advances `170105` to 82.5%
but still terminates at 5.002 cm max-axis error. Since both clips must pass,
the adaptive oracle is partial and closed, and no trained checkpoint exists.
No mass, inertia, friction, support, action scale, controller gain, reference,
or formal gate changed.

`AdaptiveMultiHorizonContactOracle` builds the exact terminal-contracted
portfolio from remaining transitions, evaluates shared H1/H5/H10 CEM
candidates, applies a formal-gate barrier, compares them with the state-only
`GateFirstHorizonSelector`, and executes only the selected first 26D action.
It has no clip identity or direct object command. The bounded sequence was
32x3 by default and used the single permitted 48x4 global upgrade; the closeout
authorizes no further MuJoCo search.

## Commands

```bash
conda run -n toporetarget-rl python scripts/rl/qualify_stage16_world_wrist.py \
  --reference .local/stage16_reference_tracking_ppo/world_wrist_references/hocap_170105.world_wrist.stage16.npz \
  --reference .local/stage16_reference_tracking_ppo/world_wrist_references/hocap_170650.world_wrist.stage16.npz \
  --object-mesh .local/stage16_reference_tracking_ppo/world_wrist_objects/hocap_170105.obj \
  --object-mesh .local/stage16_reference_tracking_ppo/world_wrist_objects/hocap_170650.obj \
  --object-dynamics-audit .local/reports/stage16_object_dynamics_audit/20260801T_object_mpc_v2/object_dynamics_audit.json \
  --scene-root .local/experiments/stage16_world_wrist_finger/qualification_replay \
  --report-root .local/reports/stage16_world_wrist_finger/qualification_replay \
  --formal-episodes 20
```

Use `--stop-after-w2` only for the isolated W2 gate; a formal oracle run
requires the explicit object-dynamics audit above.

PPO commands must include the qualifying `--oracle-report`,
`--controller-report`, and `--action-scale-report`; on the current evidence
they fail closed before training. See `train_stage16_world_wrist_ppo.py --help`
for the future-gated CLI rather than inventing a checkpoint path.

## Visualization boundary

`visualize_hocap_world_wrist_policy_mujoco.py` supports zero, fixed oracle,
adaptive-oracle action replay, and gated PPO checkpoint modes,
policies, interactive or headless modes, a fixed workspace camera,
MP4/PNG/contact-sheet output, independently encoded output FPS, and actual
MuJoCo reference-ghost, frame, axis, link, contact, force, and wrist-wrench
geometries. A HUD reports frame index, wrist/object error, reward, contact
count, termination, selected/effective horizon, candidate gate values, and
selection reason. `--kinematic-object-diagnostic` is explicitly W2-only
and cannot be described as free-object control.

The current ignored visualization bundle is under
`.local/reports/stage16_world_wrist_finger/contact_mpc_formal_selected_20260801/visual/`.
The two H=10 files have 40 and 41 real MuJoCo frames at 5 fps (8.0 s and
8.2 s), with full overlays. The first is failure evidence and the second is an
oracle pass; neither is PPO or a physically calibrated contact-support claim.
No single-clip or two-clip MuJoCo PPO video exists. MuJoCo remains available
for deterministic debugging and visualization; future PhysX policy work must
requalify its own oracle and cannot inherit this MuJoCo gate.

## Isaac Lab handoff boundary

Stage 16-C.0 adds an isolated Isaac Sim/Lab platform qualification, and C.1
separately validates Wuji/HO-Cap asset migration. C.2 then validates the custom
26-D `DirectRLEnv` on real GPU with no rollout object pose write or wrist
teleport. It does not consume a MuJoCo action trace, reward, termination, or
oracle as PhysX success. C.3 is partial because dynamic wrist tracking exceeds
its 2 cm/10 degree diagnostic bound and direct all-hand contact proof is absent;
C.4/C.5 are gate-blocked and C.6 PPO is not authorized. MuJoCo stays the CPU
correctness, deterministic-regression, contact-diagnostic, action-replay, and
interactive-visualization backend.
