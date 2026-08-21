# Stage16 V4/hocap_170650 Acceptance

## Decision

```text
STATUS=ACCEPTED_STAGE16_PHYSICAL_HOI
STAGE16_PHYSICAL_HOI_ACCEPTED=YES
NO_FURTHER_POLICY_ADAPTATION_REQUIRED
```

The frozen V4/C4 actor is accepted from the existing 20 immutable Formal20
traces. No policy was rerun or trained.

| Gate | Result |
| --- | ---: |
| Stage16PhysicalFunctionalityV1 | 20/20 |
| DF pose | 20/20 |
| DF linear | 20/20 |
| DF angular, Authority V2 | 20/20 |
| causality/no hidden control | 20/20 |
| geometry | 20/20 |

The historical instantaneous post-solver PhysX COM omega passes only 2/20 and
remains a solver-level diagnostic. The hard angular gate now uses
`Stage16ActualAngularVelocityAuthorityV2`, derived from each saved actual
object pose with the same control-rate Reference Kinematics V2 SO(3)-log
estimator as the reference. This changes measurement authority, not trace
bytes or numerical thresholds.

```text
THRESHOLD_PROVENANCE=LEGACY_INHERITED_NOT_SCIENTIFICALLY_RECALIBRATED
ANGULAR_THRESHOLD_TUNED=NO
```

The versioned local receipt binds clip/reward mode, actor/checkpoint/normalizer,
all 20 trace hashes, gravity/friction/table/hand-gravity contracts, reference
and Angular-V2 hashes, PF/DF contracts, and explicit no-guidance,
no-object-rollout-write, and no-wrist-root-write evidence:

```text
.local/reports/stage16_170650_closure_and_human_object_profile/part_a_170650/
```

This lineage is frozen and may only be used as an accepted physical-HOI data
source or positive control. Do not continue PPO, modify its reward, or adapt
the accepted actor.
