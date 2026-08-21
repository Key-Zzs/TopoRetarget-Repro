# Stage16 Actual Angular Velocity Semantics

`Stage16ActualAngularVelocityAuthorityV2` is the authority used when actual
object angular motion is compared with Reference Kinematics V2. It is an
offline measurement contract; it does not rewrite historical traces or change
any threshold.

## Provenance of the historical trace field

The rollout path is:

```text
PhysX RigidBodyView.get_velocities()[3:6]
    -> IsaacLab RigidObjectData.root_com_vel_w
    -> RigidObjectData.root_state_w[10:13]
    -> world_wrist_direct_env object_twist_world[3:6]
    -> PPO trace object_twist[3:6]
    -> exported object_angular_velocity_world
```

This value is the active rigid object's center-of-mass angular velocity,
expressed in the world frame in `rad/s`. Angular velocity is independent of the
choice of point on one rigid body, so the actor-frame pose versus COM-velocity
layout does not require a lever-arm conversion. No local/world rotation or unit
conversion is applied by the trace writer.

The environment samples pose and velocity from the same post-physics state used
to append the trace row. IsaacLab refreshes its lazy state buffer after the
final decimated physics substep. The saved velocity is therefore an
instantaneous post-solver velocity sample, while Reference Kinematics V2 uses a
centered control-rate SO(3)-log displacement with one-sided endpoints.

## Alignment result

Static, constant world-axis, rotated-body, wrap-around, world/body conversion,
same-row, and plus/minus-one-row tests pass. Source semantics select the world,
same-row interpretation before looking at episode error. Neither a frame
conversion nor a one-row shift closes the historical trace-to-pose mismatch.

Across V4/170650 Formal20, historical trace omega versus pose-derived actual
omega has mean error `0.266015 rad/s`, p95 `0.822387 rad/s`, and maximum
`1.213692 rad/s`. About `45.938%` of valid rows have pose-derived speed at most
`0.001 rad/s` while the instantaneous trace speed is at least `0.05 rad/s`.
The saved field is thus not kinematically closed to sampled pose at the
comparison bandwidth. The frozen trace lacks substep and sleep/wake telemetry,
so a more specific PhysX solver attribution is not identifiable.

```text
ROOT_CAUSE=POSE_DERIVED_REQUIRED_FOR_COMPARABLE_SEMANTICS
CONFIDENCE=HIGH
FRAME_OR_TIMESTAMP_BUG_PROVEN=NO
```

## Authority V2

Authority V2 derives actual world angular velocity from each saved actual
object quaternion with the same Reference Kinematics V2 estimator and the same
`0.05 s` control timestamps used for the reference. The operation is a
measurement-side comparison alignment, not a fitted correction to the trace:

```text
SOURCE=trace.object_pose quaternion wxyz
FRAME=WORLD
ESTIMATOR=centered SO(3)-log with one-sided endpoints
CONVERSION=NONE
HISTORICAL_TRACES_REWRITTEN=NO
ANGULAR_THRESHOLD_TUNED=NO
```

Under the inherited V1 terminal limits, Authority V2 yields 10/10 angular DF
for V4/170105 and 20/20 for V4/170650. These limits remain
`LEGACY_INHERITED_NOT_SCIENTIFICALLY_RECALIBRATED`; the authority repair does
not scientifically calibrate them.

Machine-readable provenance, per-episode aggregates, and synchronized
per-frame CSVs are generated under:

```text
.local/reports/stage16_angular_semantics_and_raw_grasp_authority/angular_semantics/
```
