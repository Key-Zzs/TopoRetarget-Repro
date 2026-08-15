# Object Guidance V1

`ObjectGuidanceContractV1` is an explicitly labelled
`ENGINEERING_EXTENSION_ASSISTED_DYNAMICS`, not author-exact TopoRetarget RL.
It is disabled by default (`guidance.mode: none`).  Enabling
`reference_wrench_v1` makes the simulation assisted, not causal physics:
`external_guidance=true`, `assisted_dynamics=true`, and `causal_physics=false`.

The world-frame guidance law consumes only Reference Kinematics V2's corrected
timestamps, pose-derived linear velocity, and pose-derived world angular
velocity:

```text
e_p = p_ref - p                 e_v = v_ref - v
a_g = Kp_p e_p + Kd_p e_v       F_g = m a_g

e_R = Log(R_ref R^T)            e_w = w_ref - w
alpha_g = Kp_R e_R + Kd_R e_w   tau_g = I_world alpha_g
```

Translation/rotation deadbands suppress tiny corrections.  Acceleration vector
norms are capped before multiplication by the actual PhysX mass/inertia, so
the force and torque are bounded without clip-specific Newton limits.  The
runtime applies the resulting world-frame wrench through IsaacLab's
instantaneous rigid-object wrench composer immediately before the PhysX step.
It never writes object pose, object velocity, or wrist-root state during a
rollout, and creates no attachment, joint, teleport, or kinematic object.

The policy action (26D), observation (764D), references, controller, and
Reward V3/V4 are untouched.  Guidance values are telemetry/evaluation only.
Each control step records the wrench, four errors, clip flags, active flag, and
derived force/torque limits.  Evaluation additionally reports impulse, work,
contact-relative ratios, and the pre-registered contact-free-guided-tracking
diagnostic; it must report `N/A` for a zero denominator.
