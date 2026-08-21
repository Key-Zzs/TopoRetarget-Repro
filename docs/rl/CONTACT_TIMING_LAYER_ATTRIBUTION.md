# Stage16 Contact Timing Layer Attribution

`Stage16ContactTimingLayerAttributionV1` is an offline diagnostic over the frozen
V4 C4 traces. It separates contact timing into three authorities:

```text
raw HOCap MANO/object
    -> geometric-retarget robot/object reference
    -> recorded PhysX named-fingertip contact
```

The raw layer uses the immutable Strict V4 source-contact mask. The retarget
layer uses `ReferenceContactContractV2.strong_contact_expected`, whose frozen
geometric threshold is `<=2 cm`. The actual layer uses named fingertip to active
object pair presence. All three layers use the existing three-control-step
persistence convention, and multi-finger readiness requires at least two
persistent fingers. LIFT is reconstructed only from `trace.reference_index`.

Runtime frames are separated from source time. One runtime control step is
`0.05 s`; raw overlay interpolation maps the same runtime indices to HOCap
source time without changing the reference or retiming the policy.

## Frozen result

| Clip | Raw ready | Retarget ready | Actual ready | LIFT | Raw margin | Retarget margin | Actual margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V4/170105 | 192 | 181 | 198 | 184 | -8 | +3 | -14 |
| V4/170650 | 136 | 129 | 162--164 | 184 | +48 | +55 | +20--22 |

Positive margins are before LIFT. At 20 Hz, 170105's retarget reference is ready
`0.15 s` before LIFT, while actual readiness is `0.70 s` after LIFT. However,
the strict raw MANO multi-finger authority is itself ready `0.40 s` after LIFT.
The retarget layer therefore advances rather than delays raw readiness by 11
frames. The actual layer is 17 frames later than the retarget layer.

The frozen decision is:

```text
CONTACT_TIMING_LAYER_ROOT_CAUSE=INCONCLUSIVE
CONFIDENCE=LOW
```

This is not a missing-data result: all three onsets are identifiable. It is a
decision-tree boundary. `RAW_TO_RETARGET_TIMING_LOSS_PRIMARY` is unsupported
because retarget timing is earlier, not later. The clean
`RETARGET_TO_PHYSICS_CONTACT_ACQUISITION_LAG_PRIMARY` predicate is also not met
because raw MANO is not grasp-ready before LIFT. The observed retarget-to-actual
lag remains real and quantified, but it cannot by itself establish that the
170105 no-lift outcome is exclusively a physics-acquisition failure.

The complete per-episode and per-finger receipts are generated under:

```text
.local/reports/stage16_contact_timing_angular_twist_pf_df/contact_timing/
```

## V2 raw-authority review

The earlier raw layer above is retained as the immutable V1 Strict-V4 timing
receipt, but Strict V4 is a reward-specific named-human-finger to named-robot-tip
target and is not a validated functional human-grasp authority. The additive
`RawHumanGraspReadinessProfileV1` now reports all-surface, region, segment,
topology, and relative-motion layers separately.

For 170105, persistent any-surface contact begins at frame 182, multi-region
contact at 190, Strict V4 target readiness at 192, and geometric opposition at
197, with LIFT at 184. Functional raw readiness remains
`NOT_IDENTIFIABLE`. Retarget-to-actual lag remains measured at 17 frames, but
the primary layer cannot be selected using an invented raw binary.

```text
CONTACT_TIMING_ATTRIBUTION_PROFILE_BASED=YES
DOES_NEW_AUTHORITY_RESOLVE_170105=PARTIALLY
CONTACT_TIMING_V2_ROOT_CAUSE=INCONCLUSIVE
CONFIDENCE=MEDIUM
FRICTION_PRIMARY=NOT_SUPPORTED
```

See [raw human grasp readiness authority](RAW_HUMAN_GRASP_READINESS_AUTHORITY.md).
This review does not modify the reference, retime contact, train a policy, or
authorize geometric or physical refinement.
