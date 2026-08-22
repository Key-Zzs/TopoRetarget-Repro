# Stage16 SourceProfileTracking Physical Refinement V1

> Historical, offline-rejected experiment. Its one-off materialization,
> validation, library, and test entrypoints were removed during physical-pipeline
> closeout; this document and its immutable receipts are retained as evidence.

## Status

```text
OFFLINE_OBJECTIVE_VALIDATION=FAIL
PROFILE_OBJECTIVE_CLASSIFICATION=PROFILE_OBJECTIVE_NOT_DISCRIMINATIVE
FINAL_CLASSIFICATION=OFFLINE_OBJECTIVE_INVALID
PPO_TRAINING_RUN=NO
```

This first bounded implementation reused `HumanObjectCouplingContactProfileV1`
without changing its raw MANO, time-map, semantic finger, or pose-derived
angular authority.  It is an additive candidate objective only; Reward V4,
physics, controller, reference, PF, DF, and the accepted 170650 lineage remain
unchanged.

## Frozen V1 objective

For the identity reference index `phi(t) = t`, with no DTW, learned time warp,
or outcome-dependent shift:

```text
L_profile = mean(component pseudo-Huber losses)
r_profile = exp(-L_profile)
```

| Channel | Source quantity | Actual PhysX quantity | Normalization | Added to reward? |
| --- | --- | --- | --- | --- |
| Activity | five persistent raw MANO regions | soft `1-exp(-named-tip-force/lambda_tip)` | dimensionless | Not promoted |
| Geometry | raw object-local contact centroid | source-active named-tip centroid in actual object-local frame | reference object-axis span | Not promoted |
| Linear coupling | pose-derived `C_v` | causal pose-derived `C_v`, never instantaneous PhysX omega | global all-source-frame p95 | Not promoted |
| Angular coupling | pose-derived `C_omega` | causal pose-derived `C_omega` | global all-source-frame p95 | Not promoted |

All channel weights and the profile reward weight are globally `1.0`.  The
`lambda_tip` scale is the existing frozen V4 scalar, not a per-object profile
weight.  Exact opposition matching and exact slip remain diagnostic-only
because the immutable actual traces do not have contact points or normals.

## Offline hard gate result

The evaluator compared 10 frozen V4/170105 C4 trajectories to the 170105
source and 20 accepted V4/170650 C4 trajectories to the 170650 source.  In the
frozen CONTACT-to-early-LIFT window, the total medians were `0.035173` for
170105 and `0.052857` for 170650.  The required direction
`170105 failed > 170650 accepted` therefore did not hold.  Contact, geometry,
and linear-coupling terms also pointed in the wrong direction; only angular
coupling had the expected direction.

The positive control is numerically sensible (`170650` p95 total loss
`0.053743 < 4.0`), but a sensible positive control alone is insufficient.
The objective cannot be added to PPO when its aggregate loss makes the known
failure look better tracked than the accepted control.

## Safety outcome

```text
170650_STATUS=ACCEPTED_STAGE16_PHYSICAL_HOI
170650_PPO_RUN=NO
BASE_V4_REWARD_CHANGED=NO
SOURCE_PROFILE_REWARD_ADDED=NO
PROFILE_WEIGHT_SWEEP_RUN=NO
LR_SWEEP_RUN=NO
EPOCH_SWEEP_RUN=NO
KL_SWEEP_RUN=NO
FIXED_PRE_LIFT_GRASP_GATE_ADDED=NO
MANUAL_GRASP_FRAME_ADDED=NO
```

No gradient sanity run, PPO update, Eval10, Confirm20, or baseline-vs-profile
training comparison was authorized. The failed objective is not a production
refinement option and has no remaining runtime dependency.

The immutable receipts and tables are under
`.local/reports/stage16_source_profile_tracking_refinement_v1/`.
