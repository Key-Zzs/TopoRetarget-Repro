# Stage16 full-gravity causal status

## Current verdict

**P3 is BLOCKED at global C2 selection. P4 is INSUFFICIENT and has not run.**
The physical route completed the C0--C2 development pilots for the frozen V3
and V4 contact-reward modes. Neither mode passed the mandatory C2 absolute
geometry gate on both clips, so no global policy was selected.

G3 is therefore recorded as `G3_PROMOTION_BLOCKED`. Its nominal full-gravity
contract and safe-state roster are retained for audit, but no rejected or
clip-specific C2 checkpoint may be run under that contract. C3, C4, P4, replay
export, and every full-gravity claim remain `NOT_RUN` or unsupported.

## Frozen causal boundary

The route retains the causal Stage16-D restrictions: no external guidance, no
invented support, no attachment shortcut, no hidden object controller, no
rollout-time object-state write, and no rollout-time wrist-root write. Gravity
and friction are stage-wide curriculum parameters, not contact-conditioned
controls.

G3 would require C4 physics (Earth-nominal gravity and nominal friction), four
replicas per retained safe state, 20 control steps, zero forbidden writes, and
the frozen runtime collision-proxy geometry contract. The prerequisite is a
single C2 policy that passes both clips.

## What would unblock P4

1. Repair the C2 absolute-geometry failure without changing the causal
   boundary or silently weakening the gate.
2. Rerun C2 for both modes and both clips, then make one global selection.
3. Run G3 for the selected mode only; a pass unlocks C3 and C4.
4. Complete the full P4 causal qualification before reporting any 1g result or
   exporting a formal replay/dataset artifact.

Historical zero-gravity qualifications are useful baselines but are explicitly
not substitutes for the unavailable full-gravity qualification.
