# Stage16 Full-Gravity Capability Closure

## Final state

`NO_SUCCESS` — all authorized technical repair and minimal adaptation branches
completed, but no lineage passed Formal20. This is a completed engineering and
evaluation closeout, not a physical-capability promotion.

The executable receipt root is
`.local/reports/stage16_full_gravity_capability_closure/`. It contains the
four-source C4 matrix, every exact PPO batch/checkpoint, isolated-process
receipts, Formal20 analysis, replay boundary, and data-export eligibility.

## Technical repair

The former monolithic sweep could time out and leave post-reset vector rows in
a trace. The repaired runner executes each condition in an isolated fresh
process group, records its own timeout receipt, and slices a replica to its
actual rollout length before checking terminal semantics. An early physical
termination is therefore retained as
`COMPLETE_DIAGNOSTIC_SWEEP_WITH_PHYSICAL_FAILURE`; it cannot be converted into
a terminal completion by rows from the next reset.

The two requested missing C4 receipts and the known-good C4 regression are all
technically complete. Frozen evaluation always has zero optimizer steps and
asserts unchanged actor and normalizer hashes.

## C4 matrix and Formal20

| Frozen source | C4 Eval10 | Decision |
| --- | --- | --- |
| V3 / hocap_170105 | 9/10 grasp, 0/10 lift | Partial; C1-only adaptation |
| V4 / hocap_170105 | 10/10 grasp, 0/10 lift | Partial; C4-only adaptation |
| V3 / hocap_170650 | 10/10 grasp, 0/10 lift | Partial; C1-only adaptation |
| V4 / hocap_170650 | 10/10 grasp, 10/10 lift | Direct frozen Formal20 |

V4/`hocap_170650` completed the held-out Formal20 without PPO, but only 2/20
episodes qualified (`SRkin=1.00`, `SRphysics=0.10`, `SRqualified=0.10`; the
required qualified rate is 0.80). It is not a successful full-gravity policy.

## Minimal-adaptation decision tree

- V3/`hocap_170105`: C1 started directly from the immutable frozen source with
  V3 reward and uniform RSI `[0,320]`. Its full 1,048,576-sample budget did not
  produce a functional Eval10 checkpoint. Stop at C1.
- V3/`hocap_170650`: C1 U2 and C2 U2 both passed frozen Eval10 and Confirm20.
  Frozen C3 then failed, and its full 4,194,304-sample C3 budget did not
  recover a functional checkpoint. C4 is blocked.
- V4/`hocap_170105`: C4 began directly from the frozen V4 source. Its full
  4,194,304-sample C4 budget did not produce a functional checkpoint.

Every training update used the authoritative baseline LR and exact PPO batch
capture. No rejected 0.5x actor LR, reward, KL, epoch, seed, or held-out sweep
was used. Eval10 was frozen Frame0 evaluation; Confirm20 followed each real
functional checkpoint.

## Stop boundary

No policy passed Formal20, so there is no success data export and no success
replay command. Retained simulations, checkpoints, and exact batches remain
diagnostic evidence only. The only current action is:

```text
STOP_NO_FURTHER_PPO_OR_SWEEP_AUTHORIZED
```

Any new training or success claim requires a separately authorized protocol.
