# Stage16 Angular-Twist Audit

`Stage16AngularTwistAuditV1` explains the V4/170650 Formal20 angular residual
without changing its inherited threshold. For every saved episode it compares:

1. recorded PhysX object angular velocity;
2. angular velocity derived from recorded actual object orientation;
3. frozen Reference Kinematics V2 angular velocity.

Actual and reference pose-derived velocities use the exact Reference Kinematics
V2 SO(3)-log world-frame estimator, the frozen `0.05 s` runtime timestamps, and
the same centered/one-sided boundary handling. Euler-angle differentiation is
not used.

## Frozen result

The reference estimator is internally consistent: stored reference omega and
pose-derived reference omega differ by only about `4.6e-7 rad/s` mean and
`9.6e-7 rad/s` p95. This rules out a Reference V2 boundary or discretization
artifact as the primary explanation.

Recorded PhysX omega does not match actual pose-derived omega:

| Metric | Formal20 aggregate |
| --- | ---: |
| trace-vs-pose mismatch mean | 0.266015 rad/s |
| trace-vs-pose mismatch p95 | 0.822387 rad/s |
| trace-vs-pose mismatch max | 1.213692 rad/s |
| trace Delta omega mean | 0.270014 rad/s |
| pose-derived Delta omega mean | 0.023421 rad/s |
| pose-derived Delta omega p95 | 0.069188 rad/s |
| terminal trace pass under V1 | 2/20 |
| terminal pose-derived pass under V1 | 20/20 |

The frozen attribution is:

```text
DOES_TRACE_OMEGA_MATCH_POSE_DERIVED_OMEGA=NO
ANGULAR_TWIST_ROOT_CAUSE=ANGULAR_VELOCITY_MEASUREMENT_SEMANTICS_MISMATCH_PRIMARY
ANGULAR_THRESHOLD_TUNED=NO
```

Trace exceedances contain both isolated segments and persistent runs, with the
largest trace error in late motion. Those structures describe the recorded
velocity field; they are not evidence of actual rotational wobble when the
same recorded orientation yields much smaller pose-derived residuals. The
object-minus-wrist angular-twist field is retained as a labeled proxy, but its
trace-based value cannot override the primary measurement-consistency failure.

Per-frame CSVs, phase summaries, estimator consistency, and replay selections
are generated under:

```text
.local/reports/stage16_contact_timing_angular_twist_pf_df/angular_twist/
```

## Authority V2 closeout

The measurement-semantics action is complete. Static IsaacLab/PhysX provenance
shows that the historical field is world-frame instantaneous COM angular
velocity from `RigidBodyView.get_velocities()`, sampled after the final physics
substep. It is not kinematically closed to control-rate sampled pose; no frame
conversion or plus/minus-one-row shift repairs that mismatch.

`Stage16ActualAngularVelocityAuthorityV2` therefore derives actual omega from
the saved actual object quaternion with the same world-frame, control-rate
SO(3)-log estimator used by Reference Kinematics V2. This changes no historical
trace byte and tunes no threshold. V4/170650 is 20/20 under Authority V2 versus
2/20 under the legacy instantaneous trace field. See [actual angular velocity
semantics](ACTUAL_ANGULAR_VELOCITY_SEMANTICS.md).

The next action is
`NEXT_REQUALIFY_170650_WITH_ANGULAR_AUTHORITY_V2`; this audit does not add a
reward or authorize training.
