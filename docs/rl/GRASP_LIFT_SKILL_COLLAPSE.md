# Stage16 Grasp-Lift Skill Collapse

The frozen localization establishes a one-update U25-to-U26 loss of persistent
grasp and lift. Its follow-up policy-preservation shadow search repaired the
actor-only/critic-baseline harness and selected a 0.50x actor-LR candidate.
The candidate preserves 10/10 frame-0 grasp/lift and U25 GRASP-reset lift,
while its U25 CONTACT reset remains a documented limitation.

The shadow result was a controlled single-update candidate only, not a
longitudinal preservation result. Its historical reproduction remains
attribution evidence only.

The subsequent live full-C0 V3/`hocap_170105` validation completed all 26
updates. It distinguishes the two claims: the single U25-to-U26 shadow candidate
retained 10/10 grasp/lift, but the live 0.50x actor-LR candidate lost both at
U17 / 696,320 samples and remained at 0/10 through U26 and endpoint Eval20.
The original 1.0x C0 lineage was still 10/10 at U25 and only collapsed at U26.
Thus the live result is `CANDIDATE_REGRESSION`,
`STATUS=SHADOW_ONLY_NOT_SUFFICIENT`, not a repair of the grasp/lift forgetting
mechanism. It neither switches the PPO default nor authorizes C1. The sole
next action is `NEXT_UPDATE_DEPTH_POLICY_PRESERVATION_ABLATION`; the retained
0.98/0.25 saturation thresholds continue to be telemetry warnings rather than
curriculum stop gates. See
[full C0 longitudinal validation](CONTACT_PRESERVING_FULL_C0_VALIDATION.md).
