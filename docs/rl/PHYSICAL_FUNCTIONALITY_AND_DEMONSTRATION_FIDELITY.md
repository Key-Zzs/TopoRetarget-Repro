# Physical Functionality and Demonstration Fidelity

Stage16 now reports two orthogonal profiles:

```text
PF: Can the robot physically perform the interaction?
DF: Does the resulting physical motion remain faithful to the demonstrated/reference motion?
```

In particular:

```text
PF != DF
```

`Stage16PhysicalFunctionalityV1` is based only on recorded PhysX outcome and
safety: causal execution, safe geometry, bounded actions, multi-finger
persistent grasp readiness before/on LIFT, at least 5 cm of actual object lift,
and absence of hidden guidance/object/wrist-root writes. Source-required named
finger recall, table support transfer, and relative hand-object coupling are
reported supporting metrics because no additional frozen hard threshold exists.

## PF V2 causal-lift audit (frozen additive authority)

An additive `Stage16PhysicalFunctionalityV2` proposal moved reference-LIFT
contact timing into `DFInteractionTimingV1` and evaluated actual support-free
lift from the inherited 5 cm displacement event. It did not modify PF V1.
Its table-support proxy is binary rather than an exact wrench/slip measurement.
The table ContactSensor is independent from the hand-pair force stream: each
historical accepted 170650 trace has a retained reset support sample before
release. The prior 0/20 support result was a mask-authority bug, now repaired.

The corrected positive-control audit is
`PF_V1_PRELIFT_GATE_PARTIALLY_OVERCONSTRAINED`. U10/170105's no-step Eval20
and U11 Confirm20 both give PF V2 causal lift 20/20 while PF V1 stays 0/20
because its pre-reference-LIFT timing gate remains immutable. A symmetric
170650 experimental continuation is stable at U2 Eval20 (PF V1/PF V2/DF all
20/20), but is non-monotonic (U8 PF V2=0/10, U10 Eval10=10/10); the frozen
historical accepted actor is unchanged.
See [PF V2 causal-lift audit](PHYSICAL_FUNCTIONALITY_V2_CAUSAL_LIFT.md).

`Stage16DemonstrationFidelityV1` reports separate dimensions rather than an
unvalidated total boolean:

- `DF_pose` uses frozen V2 `Er/Et/Ej/Eft` thresholds;
- `DF_linear` reports reference-relative linear velocity and the inherited V1
  terminal result;
- `DF_angular` reports trace-based and pose-derived angular velocity, estimator
  consistency, phase structure, and inherited V1 terminal results.

The linear and angular numerical limits are
`LEGACY_INHERITED_NOT_NEWLY_VALIDATED`. A pass or failure under those limits is
not presented as a scientifically calibrated fidelity threshold.

## Four possible profiles

| PF | DF | Meaning |
| --- | --- | --- |
| PASS | PASS | physically executable and faithful |
| PASS | DEGRADED | physically works but motion differs from the demonstration |
| FAIL | PASS | tracks the reference but cannot complete the physical interaction |
| FAIL | DEGRADED | neither physically functional nor faithful |

These are conceptual quadrants; they do not hard-code a clip result.

## Frozen V4 receipts

| Metric | V4/170105 Eval10 | V4/170650 Formal20 |
| --- | ---: | ---: |
| PF | 0/10 | 20/20 |
| DF pose | 0/10 | 20/20 |
| DF linear under current V1 | 0/10 | 20/20 |
| DF angular, trace under current V1 | 10/10 | 2/20 |
| DF angular, pose-derived under current V1 | 10/10 | 20/20 |
| legacy SRqualified | 0/10 | 2/20 |
| SR dynamic V1 | 0/10 | 2/20 |

PF/DF separation changes the interpretation of 170650: it is physically
functional and pose/linear faithful. The legacy instantaneous omega field fails
its inherited angular gate, but the comparable-semantics
`Stage16ActualAngularVelocityAuthorityV2` passes 20/20. Accordingly, 170650 is
accepted for physical-HOI evidence under PF, pose, linear V1, and angular
Authority V2, with the explicit caveat that the numerical velocity limits are
still inherited and not scientifically recalibrated. For 170105, later contact
without lift remains a PF failure regardless of its 10/10 Authority V2 angular
result.

The complete receipt is ignored local evidence under:

```text
.local/reports/stage16_contact_timing_angular_twist_pf_df/pf_df/
```

The additive Authority V2 comparison is under:

```text
.local/reports/stage16_angular_semantics_and_raw_grasp_authority/pf_df/
```

The formal versioned acceptance receipt is under:

```text
.local/reports/stage16_170650_closure_and_human_object_profile/part_a_170650/
```

It records `ACCEPTED_STAGE16_PHYSICAL_HOI` only after PF, pose, linear,
Authority-V2 angular, causality, and geometry each pass 20/20. The accepted
V4/hocap_170650 lineage is frozen and requires no further policy adaptation.
See [Stage16 170650 acceptance](STAGE16_170650_ACCEPTANCE.md).

Neither PF nor DF modifies Evaluation Suite V2, historical `SRphysics`, or
Stage16 Dynamic Physical Qualification V1.
