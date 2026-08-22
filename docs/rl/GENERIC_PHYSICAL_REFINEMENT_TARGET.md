# Generic Stage16 Physical Refinement Target

## Decision

```text
PRIMARY_TARGET=SOURCE_PROFILE_TRACKING
CONFIDENCE=MEDIUM
FRICTION_PRIMARY=NOT_SUPPORTED
NEXT_IMPLEMENT_OBJECT_AGNOSTIC_PHYSICAL_REFINEMENT_V1:SOURCE_PROFILE_TRACKING
```

This is a historical objective-selection record. It does not implement a
reward or authorize training.

## V1 implementation result

`Stage16SourceProfileTrackingV1` was subsequently implemented as an
offline-only hard gate using the selected activity, object-local geometry, and
pose-derived coupling channels.  The gate was numerically finite and did not
regress the 170650 positive control, but it was
`PROFILE_OBJECTIVE_NOT_DISCRIMINATIVE`: 170105's combined
CONTACT-to-early-LIFT loss was lower than 170650's.  No PPO update was
authorized and V4 is unchanged.  The exact bounded receipt is in
`docs/rl/SOURCE_PROFILE_TRACKING_REFINEMENT.md` and the ignored local report
root named there. Its one-off runtime implementation was removed during
physical-pipeline closeout after this terminal gate; the record remains for
scientific provenance, not as a production target.

## Why source-profile tracking

For 170105, raw any-surface contact begins at frame 182, only two frames before
LIFT 184; multi-region and opposing topology form at 190 and 197. Retarget
persistent multi-tip readiness exists by 181, but actual persistent multi-tip
contact appears at 198, a 17-frame/0.85-s delay. The object never transfers to
a successful lift. A fixed “grasp before LIFT” target would distort the source
human style.

For the positive 170650 control, raw contact starts at 109, multi-region and
opposition are established by 136/140, retarget readiness is 129, actual
readiness is 162, and all 20 episodes transfer/lift successfully. Contact
acquisition lag alone therefore does not distinguish success; the full
time-varying source interaction does.

Let

```text
I_source(t) = [region activity or object-local contact distribution,
               geometric opposition/spread when authoritative,
               dimensionless relative-linear coupling,
               dimensionless relative-angular coupling].
```

PhysX computes the corresponding `I_robot(t)`. The proposed single objective
family is

```text
L_profile = integral rho(W [I_robot(phi(t)) - I_source(t)]) dt,
```

where `phi` is source/reference phase alignment, `rho` is one robust vector
loss, and `W` uses global dataset-level normalization rather than per-object
tuning. Support-transfer success is retained only as a supporting functional
outcome. Exact slip cannot be selected as primary until object-local contact
point tracks, normals, effective friction, and normal/tangential force
authority exist.

For 170105's retarget-ready-to-actual-persistent window (frames 181--198), the
actual median pose-derived relative speeds are `0.00774 m/s` and
`0.0498 rad/s`; actual any-hand contact first appears at 189 and persistent
multi-tip contact at 198. The available evidence therefore supports tracking
the full phase-conditioned interaction profile, but does not isolate friction,
exact slip, or one coupling threshold as the sole cause.

The objective is object-agnostic because it uses relative poses, normalized
motion, phase, and geometry distributions instead of clip IDs, fixed object
mass, manually raised friction, or object-specific reward weights.

```text
FIXED_PRE_LIFT_GRASP_GATE_RECOMMENDED=NO
PER_OBJECT_REWARD_TUNING_REQUIRED=NO
PER_OBJECT_FRICTION_TUNING_REQUIRED=NO
MANUAL_GRASP_FRAME_LABEL_REQUIRED=NO
```

The frozen candidate matrix and complete machine decision are under
`.local/reports/stage16_170650_closure_and_human_object_profile/refinement_decision/`.
