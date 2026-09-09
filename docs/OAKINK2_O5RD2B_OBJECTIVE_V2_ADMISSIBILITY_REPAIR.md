# OakInk2 O5R-D2B ObjectiveV2 admissibility repair

O5R-D2B audited the continuity semantics used by the D2A Candidate B search. The
`0.01 m`, `5 deg`, and `0.05 rad` values in `continuous.py` normalize correction
from a transported prediction and also bounded the optimizer trust region. Their
use sites do not establish physical, robot-control, Semantic-V1, or actual
trajectory hard-limit authority.

Frozen `RetargetSemanticValidityV1` instead gates actual frame-to-frame wrist
motion at `0.05 m` translation and `90 deg` rotation. It defines no hard
finger-q step threshold. Predictor correction and actual trajectory motion are
therefore distinct quantities.

The D2A HIGH window had 10 usable states and 10 states rejected only by the
predicted finger-correction budget. All 20 passed the actual Semantic-V1
translation and rotation continuity limits. This evidence justified Candidate
B2, which preserves Candidate B's interaction hinge and demotes prediction
correction to a soft preference/diagnostic. It did not justify Candidate C,
whose branch requires true temporal-continuity failures.

Candidate B2 was evaluated sequentially on the already-consumed HIGH, MID, and
LOW development windows. MID and LOW met the interaction target, but HIGH had
only 18/20 technical completions and remained above the target at p95. LOW also
had one technical noncompletion. The frozen development budget was not
increased.

Consequently:

```text
OBJECTIVE_V2_PATH=CANDIDATE_B2
RETARGET_OBJECTIVE_V2_DESIGN_STATUS=NO_CANDIDATE_READY
RETARGET_OBJECTIVE_V2_SHA256=null
OBJECTIVE_V2_EXECUTION_CONTRACT_SHA256=null
SPARSE_VALIDATION_V2=NOT_RUN
WINDOW_VALIDATION_V2=NOT_RUN
DEV2_FRAME0_HARD_CONTROL=NOT_RUN
FULL_DEV2_COMPUTE_AUTHORIZED=NO
DEV2_FULL_PRODUCTION_SOLVE_COUNT=0
```

The authoritative machine-readable evidence is under
`.local/reports/oakink2_o5rd2b_objective_v2_certification_v1/`.

