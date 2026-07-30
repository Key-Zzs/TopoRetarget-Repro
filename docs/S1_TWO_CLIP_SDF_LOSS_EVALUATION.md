# S1 two-clip E0 versus dense SDF loss

The reproducible run is described by
`configs/experiments/s1_sdf_penetration_loss_v1.yaml` and is limited to:

| ID | sequence | native half-open range | evaluated frames |
|---|---|---:|---:|
| G1 | `s1/airplane_lift` | `[240, 300)` | 60 |
| G2 | `s1/apple_eat_1` | `[212, 272)` | 60 |

Both clips use the same right-hand Arti-MANO, frame/bone/solver/query and
execution profiles. E0 is `lambda_sdf=0`; S1 uses one unified value selected
from `{0.01, 0.1, 1.0}` after the fixed prescreen frames `0, 29, 59` plus
E0-derived extrema have been frozen. G1 and G2 are never allowed to select
different lambdas.

For every candidate and final run, the report records solver status, optimizer
convergence, active-set feasibility, joint/slack bounds, full 512-point signed
distance audits, minimum signed distance, negative-sample count/fraction,
geometry-balanced SDF energy, fallback-gradient count, runtime, and input
artifact hashes. Status 9 is rejected under the frozen v3 policy.

The final decision is one of:

- `S1_SDF_PENETRATION_LOSS_ACCEPTED`: a unified lambda passes the hard and
  regression gates;
- `S1_SDF_PENETRATION_LOSS_INACTIVE_EQUIVALENT`: E0 has no signal and S1 is
  numerically equivalent;
- `S1_SDF_PENETRATION_LOSS_REJECTED_NO_SIGNAL`: E0 has no signal but S1 moves;
- `S1_SDF_PENETRATION_LOSS_REJECTED`: a candidate violates a hard/regression
  gate;
- `S1_BLOCKED_BY_FORMAL_PIPELINE_FAILURE`: the required run or audit did not
  complete.

No manual acceptance, G3/G4, ContactPose, raw-data modification, or main
worktree artifact reuse is part of this stage.

## Relationship to S1.1

S1.1 is a later diagnostic lane and does not revise this frozen two-clip
baseline. It excludes G1/G2 from source selection, preserves this artifact
tree, and records any signal-rich result under its own experiment root. See
`docs/stages/S1_1_SIGNAL_RICH_GRAB.md` for the dependency chain.
