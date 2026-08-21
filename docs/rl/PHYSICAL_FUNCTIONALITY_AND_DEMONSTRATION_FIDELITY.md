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
functional and pose/linear faithful, while the recorded omega field fails its
inherited angular gate. The angular audit shows that this failure is dominated
by measurement semantics rather than actual pose-derived rotational motion.
For 170105, later contact without lift remains a PF failure regardless of any
individual angular pass.

The complete receipt is ignored local evidence under:

```text
.local/reports/stage16_contact_timing_angular_twist_pf_df/pf_df/
```

Neither PF nor DF modifies Evaluation Suite V2, historical `SRphysics`, or
Stage16 Dynamic Physical Qualification V1.
