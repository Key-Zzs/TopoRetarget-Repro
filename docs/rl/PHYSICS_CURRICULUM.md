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
