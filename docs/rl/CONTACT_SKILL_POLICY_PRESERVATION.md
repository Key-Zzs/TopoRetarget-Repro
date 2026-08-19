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

## Full C0 longitudinal result

The preceding result is a **single-update exact-batch shadow result**, not a
claim about a live C0 training trajectory. The required live V3/
`hocap_170105` C0 validation has now resumed from the authoritative U12
checkpoint, completed all 1,048,576 C0 samples, wrote an exact batch and
Eval10 receipt for every update, and added an endpoint Eval20 plus headless
replay validation.

The 0.50x actor-LR candidate held 10/10 persistent grasp and lift through
U16, then fell to 0/10 for both at U17 (696,320 samples) and remained there
through U26; endpoint Eval20 was 0/20 for both. By comparison, the frozen 1.0x
C0 lineage retains 10/10 at U25 (1,024,000 samples) and collapses at U26.
The fail-closed classification is therefore `CANDIDATE_REGRESSION`, with
`STATUS=SHADOW_ONLY_NOT_SUFFICIENT`, rather than a preservation result. It does
not authorize C1, a production-default switch, or a claim of controller
regression. The next allowed action is
`NEXT_UPDATE_DEPTH_POLICY_PRESERVATION_ABLATION`.

The evidence, full longitudinal curve, baseline comparison, replay commands,
and best-checkpoint receipt are recorded in
`.local/reports/stage16_contact_preserving_full_c0_validation/` and summarized
in [full C0 longitudinal validation](CONTACT_PRESERVING_FULL_C0_VALIDATION.md).
