# Stage16 Contact-Skill Policy Preservation Ablation

The frozen U25-to-U26 exact-batch shadow replay is numerically equivalent to
the canonical destructive update and reproduces the frame-0 grasp/lift
collapse. A paired shadow implementation advances the baseline critic at every
minibatch and exports that exact final critic state for each actor-only
candidate.

Stage-start KL anchoring is fail-closed: its gradient-ratio calibration is
undefined at an identical U25 anchor because the first-order KL gradient is
The 0.50x and 0.25x nonzero actor-LR candidates both retain 10/10 frame-0
persistent grasp and lift. The nearest-baseline 0.50x candidate is selected as
`Stage16ContactSkillPolicyPreservationV1`; it is opt-in and does not switch
the formal-training default. Its U25 GRASP reset is also 10/10 lift, while its
U25 CONTACT reset is a documented 0/10 lift limitation. A continuous C0 live
verification is required next.
