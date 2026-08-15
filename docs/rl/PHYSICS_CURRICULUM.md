# Stage16 gravity and friction curriculum

The physical curriculum is an engineering continuation of the causal Stage16-D
baseline. It does not change the source-derived collision geometry, controller,
mass, inertia, damping, restitution, self-collision policy, or causal-write
prohibitions while it changes gravity and friction by stage.

| Stage | Gravity scale | Friction scale | Role |
| --- | ---: | ---: | --- |
| C0 | 0.00 | 2.00 | Contact-ready physical pilot |
| C1 | 0.25 | 1.75 | Intermediate pilot |
| C2 | 0.50 | 1.50 | Global reward-mode selection |
| C3 | 0.75 | 1.25 | Post-G3 training stage |
| C4 | 1.00 | 1.00 | Nominal full-gravity training stage |

The curriculum is progress-driven only: no contact-triggered gravity or
friction changes and no per-episode physics override are permitted.

## Causal curriculum execution

Each reward-mode × clip lineage runs continuously from its frozen zero-g
checkpoint through C0, C1, C2, C3, and C4. Promotion is determined only by the
planned sample budget completing with finite executable PPO state. Saturation,
KL/clip diagnostics, interaction, reference geometry, penetration, and
Evaluation Suite metrics are recorded as warnings or final outcomes; they are
not curriculum stop gates. The inferred planar support is active from episode
start through terminal and naturally loses object contact after lift.

P3-B.6 requalified the physical reference mask, finite inferred support, and
support-aware RSI bank over all 321 frames of both HOCap clips. The formal
reference geometry gate still fails, and joint zero replay is not authorized
after runtime joint-limit terminations. The frozen decision is
`P3_RESTART_BLOCKED_REFERENCE_GEOMETRY`; PPO gravity training remains not run.
See [physical scene and RSI requalification](PHYSICAL_SCENE_RSI_REQUALIFICATION.md).

## P3-B.5 geometry attribution

Frozen A/B/C/D gravity--friction counterfactuals established that the selected
C2 failures already violate the absolute hand--object geometry gate at reset
frame 0. The result holds for saved-action and deterministic frozen-policy
replay. The current attribution is therefore `RESET_GEOMETRY_PRIMARY`, not a
gravity, friction, or policy-reaction primary cause. The only permitted next
task is `NEXT_REBUILD_PHYSICAL_SAFE_RSI_BANK`; C2 must not be retrained until
that rebuilt reset bank is formally geometry-qualified.

## C4 evaluation boundary

C4 is nominal full gravity and nominal friction. Every lineage receives the
frozen deterministic 20-episode evaluation and simulation-data export,
including failed, dropped, penetrating, and no-contact episodes. Evaluation
Suite V2 thresholds remain frozen for reporting only; no performance outcome
can suppress C3/C4 execution or export.
## P3-B.7 reset boundary

P3 C0--C2 may start only from `Stage16EarlyTableResetPoolV1`: a continuous
`PRE_CONTACT`, finite-table-supported, exact-geometry-safe window qualified
under 1g dynamics.  The table stays active for every curriculum stage; gravity
and friction still follow the fixed stage schedule for the whole episode.
Whole-reference geometry is diagnostic, while hard-reset and actual-rollout
geometry are hard gates.
