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

## Audit result: not promoted

The frozen re-evaluation found all of the following:

| Trace set | PF V1 | Physical lift | Causal lift | Support proxy | PF V2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical 170105 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| 170105 U9 | 0/10 | 9/10 | 9/10 | 9/10 | 9/10 |
| 170105 U10 | 0/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| Historical accepted 170650 | 20/20 | 20/20 | 20/20 | 0/20 | 0/20 |

The accepted 170650 formal traces contain no observed table-support frame
before their first recorded support-free state. Under a fail-closed support
transfer proxy, this is `NOT_IDENTIFIABLE`, not evidence that transfer
occurred. Therefore the audit classification is
`PF_V2_SEMANTICS_INVALID` with high confidence. PF V2 is not an acceptance
authority, no U10 Eval20 was run, and neither requested new PPO lineage was
authorized.

The ignored local audit receipt is:

```text
.local/reports/stage16_pf_v2_causal_lift_and_symmetric_ppo/
```

## Consequence

The audit supports the narrower observation that reference-LIFT timing and
actual 5 cm lift are distinct events. It does not yet validate a replacement
physical-functionality gate across the accepted positive control. Any future
revision must establish support-transfer observability prospectively, without
rewriting the frozen traces or selecting a threshold from U9/U10 outcomes.
