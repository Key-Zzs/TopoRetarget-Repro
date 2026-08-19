# Stage16 Frozen Source Policy Gravity/Friction Sweep

## Status

`TECHNICALLY_INCONCLUSIVE` — this is not a policy-wide physical failure and it
does not authorize PPO training. The executable evidence is rooted at
`.local/reports/stage16_frozen_source_policy_gravity_sweep/`.

The evaluation freezes all four selected zero-gravity source actors and their
normalizers. It performs no PPO collection, actor/critic update, normalizer
update, or optimizer step. Every completed receipt asserts the actor and
normalizer hashes before and after the physical trajectory are identical.

## Physics contract

This is the authoritative C0--C4 **gravity/friction** curriculum, not a
pure-gravity ablation:

| Stage | Object gravity | Object/table friction |
| --- | ---: | ---: |
| C0 | 0.00x | 2.00x |
| C1 | 0.25x | 1.75x |
| C2 | 0.50x | 1.50x |
| C3 | 0.75x | 1.25x |
| C4 | 1.00x | 1.00x |

The wrist is fixed, hand gravity remains disabled, objects are dynamic, and
the inferred table is active. Each valid condition has ten Frame0 vectorized
replicas and ten separate 321-frame final trace files with the matched named
seed manifest.

## Evidence result

Fourteen of twenty conditions qualified (140 of the required 200 trajectories).
Six are explicitly `TECHNICALLY_INCONCLUSIVE`, never recast as policy failures:

- V4/170105: C2 and C4 had physical-rollout timeouts before capture.
- V3/170650: C1 and C4 timed out before capture; C2 captured a trace but failed
  the required `TERMINAL` reference-phase contract.
- V4/170650: C1 timed out before capture.

The completed C4 results establish two important facts:

- V3/170105 retains persistent grasp in 9/10 episodes at C4 but has 0/10
  lifts, so it is `PARTIALLY_FUNCTIONAL` rather than functional.
- V4/170650 is 10/10 persistent grasp and lift at C4, proving a frozen source
  actor can execute the full-gravity physical trajectory without PPO adaptation.

Consequently the observed C4 capability is
`PARTIAL_FULL_GRAVITY_CAPABILITY`, but the four-lineage global decision remains
`TECHNICALLY_INCONCLUSIVE` until the missing C4 evidence is repaired. This is
why the sole next action is
`NEXT_REMEDIATE_TECHNICAL_ROLLOUT_TIMEOUTS_THEN_REEVALUATE_INCONCLUSIVE_LINEAGES`.

`PPO_TRAINING_RUN=NO` and `PPO_OPTIMIZER_STEP=0` throughout. If a later
adaptation is authorized, the completed V3/170105 curve supports starting only
from its last fully functional C0 actor rather than restarting all lineages.

## Geometry and controller evidence

The collision audit retains exact python-fcl queries for all pairs not proved
separated. Conservative authored-proxy-centre spheres and exact world AABBs
only omit pairs mathematically proven disjoint; they do not alter a penetration
threshold or classify a touching pair as separated. For colliding pairs the
same FCL collision MTD is used directly; non-colliding pairs retain FCL distance
queries.

All completed conditions have wrist command-to-actual rotation under 0.005 rad,
so `CONTROLLER_REGRESSION_WITH_GRAVITY=NO`. The available runtime trace has an
object/table signal but lacks an exact hand/table pair trace:
`hand_table_max_mm=NOT_IDENTIFIABLE_WITH_CURRENT_TABLE_TRACE`, never a pass.

## Replay

Headless replay was actually validated for the qualified C4 traces:

- V3/170105/C4, episode 00
- V4/170650/C4, episode 00

The two unqualified C4 lineages intentionally have no replay command presented
as a validation result. The report directory contains the exact headless
receipts and GUI commands.
