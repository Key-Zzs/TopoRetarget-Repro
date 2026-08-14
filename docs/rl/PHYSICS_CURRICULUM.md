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
| G3 | 1.00 | 1.00 | Full-gravity promotion diagnostic |
| C3 | 0.75 | 1.25 | Post-G3 training stage |
| C4 | 1.00 | 1.00 | Nominal full-gravity training stage |

The curriculum is progress-driven only: no contact-triggered gravity or
friction changes and no per-episode physics override are permitted.

## C2 global selection

Both frozen contact reward modes are evaluated on both clips at C2. Selection
is global, not clip-specific. A candidate must pass the absolute geometry and
causal-controller safety conditions for both clips. A tied or rejected mode is
not eligible for C3, C4, G3, or P4.

The current C2 selection rejected both modes at the absolute geometry gate.
Consequently the current status is **P3 BLOCKED at C2 selection**. This is a
safety conclusion, not a claim that either policy has been physically promoted.

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

## G3 and P4 boundary

G3 requires the selected global C2 policy, nominal C4 physics, four replicas
per retained safe state, and 20 control steps. It verifies finite execution,
no systematic joint-limit or actuator/solver failures, zero forbidden writes,
and the frozen collision-proxy geometry contract. It is deliberately placed
between C2 and C3.

If no global C2 mode is selected, G3 must emit an upstream-blocked receipt and
must not run a rejected policy. C3/C4 and P4 are then `NOT_RUN`; neither a
zero-gravity baseline nor a partial pilot may be presented as a 1g result.

P4, if unlocked, is a 20-episode-per-clip full-gravity causal qualification
with no support injection or external guidance. Its milestone target is
`SRqualified >= 0.8` on each clip.
