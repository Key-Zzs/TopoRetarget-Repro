# Stage16 PF V2 Causal Lift Audit

`Stage16PhysicalFunctionalityV2` is an additive, unpromoted evaluator
proposal. It does not modify `Stage16PhysicalFunctionalityV1`, historical PF
receipts, or the accepted 170650 lineage.

## Intended separation

The proposal separates physical functionality from interaction timing:

```text
PF V2: actual support-free lift with causal hand-object interaction
DFInteractionTimingV1: contact and consolidation relative to reference LIFT
```

Consequently, persistent multi-finger contact after reference `LIFT` is a
timing diagnostic, not a PF V2 hard gate. The proposal instead identifies
`ActualLiftOnset` from the inherited 5 cm displacement threshold, persistent
support absence, and positive pose-derived vertical velocity. It then requires
observed hand contact, persistent multi-finger contact, and finite
pose-derived relative motion through that actual event.

`SupportTransferProxyV1` uses recorded binary table contact. It is not an
exact normal-wrench or surface-slip measurement. Exact support transfer is
therefore never claimed from these traces.

## Frozen audit and bounded result

The frozen re-evaluation found all of the following:

| Trace set | PF V1 | Physical lift | Causal lift | Support proxy | PF V2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical 170105 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| 170105 U9 | 0/10 | 9/10 | 9/10 | 9/10 | 9/10 |
| 170105 U10 | 0/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| Historical accepted 170650 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 |

The historical 170650 formal traces do contain the required support evidence:
each has `table_object_contact=True` at its reset sample and `False` after
release. The prior 0/20 conclusion was an evaluator bug: it applied the
frame-zero-invalid **hand-object pair-force** mask to the independent table
ContactSensor stream. The V2 contract now retains a recorded reset table sample
under its own validity rule. The replay's table only visualizes the frozen
support proxy; it does not produce or recompute the contact telemetry.

The corrected audit is
`PF_V1_PRELIFT_GATE_PARTIALLY_OVERCONSTRAINED` at medium confidence. The V2
contract was frozen before training. Its constants remain the inherited 5 cm,
three-control-step, two-finger rule; it makes no exact wrench-transfer or
surface-slip claim and never changes PF V1.

The zero-optimizer U10 Eval20 gave PF V2/physical lift/causal lift/support
transfer/sustained coupling = 20/20. A bounded, symmetric C4 grouped-reward/RSE
continuation then used 1,024 environments and 40 rollout steps:

| Experimental lineage | New updates / samples | Selected Eval20 | PF V1 | PF V2 | DF pose / linear / angular V2 |
| --- | ---: | --- | ---: | ---: | ---: |
| 170105 U10 → U11 | 1 / 40,960 | same-checkpoint Confirm20 | 0/20 | 20/20 | 20/20 / 20/20 / 20/20 |
| historical 170650 → U1--U10 | 10 / 409,600 | best observed U2 | 20/20 | 20/20 | 20/20 / 20/20 / 20/20 |

170105 U11 stopped because its Eval10 PF V2 was 10/10. PF V1 remains 0/20 as
required: the old reference-LIFT timing gate is retained rather than silently
relabelled as a V2 physical failure. The 170650 experimental continuation
was non-monotonic after U2: U8 PF V2=0/10, then U10 recovered to Eval10 10/10.
U2 is the first maximum-score checkpoint and is not a replacement for the
frozen historically accepted actor.

The ignored local audit receipt is:

```text
.local/reports/stage16_pf_v2_causal_lift_and_symmetric_ppo/
```

## Consequence

PF V2 validates the narrower causal-lift interpretation across both the frozen
positive control and the U11 170105 confirmation, while PF V1 remains an
immutable historical authority. The only next action is to diagnose the 170650
continuation instability without threshold changes, tuning, or a new sweep.
