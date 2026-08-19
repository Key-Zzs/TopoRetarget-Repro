# Stage16 Contact-Preserving Full C0 Longitudinal Validation

The opt-in `Stage16ContactSkillPolicyPreservationV1` candidate uses the frozen
0.50x actor learning rate with the baseline critic learning rate. This document
records the required live V3/`hocap_170105` C0 validation; it supersedes no
historical result and does not change a production default.

## Protocol and evidence

The run resumed from the authority checkpoint specified by the candidate
contract, completed 26 C0 updates and exactly 1,048,576 stage samples. Every
update has a checkpoint, exact PPO batch, train receipt, and deterministic
frame-0 Eval10. The endpoint is an independent Eval20 at U26 and has a finite
headless replay receipt. All outputs are local and untracked under
`.local/runs/stage16_contact_preserving_full_c0_validation/` and
`.local/reports/stage16_contact_preserving_full_c0_validation/`.

## Result

Source Eval10 is 10/10 persistent grasp and lift. Candidate U1--U16 are also
10/10 for both measures. U17 (696,320 samples) is the first 0/10 grasp and
0/10 lift result, and U17--U26 remain 0/10. The U26 endpoint Eval20 is 0/20
for both measures. The trajectory and endpoint have no observed runtime or
finite-tracking controller regression.

The frozen 1.0x C0 comparison is 10/10 at U25 (1,024,000 samples) and 0/10 at
U26 (1,048,576 samples). Therefore the 0.50x actor-LR candidate collapses
earlier than the historical lineage rather than delaying or preventing the
failure.

| Classification | Candidate status | Production default | C1 |
| --- | --- | --- | --- |
| `CANDIDATE_REGRESSION` | `SHADOW_ONLY_NOT_SUFFICIENT` | unchanged | not started |

The best evaluated candidate checkpoint is U1 at 40,960 samples, with 10/10
persistent grasp and lift; its SHA256 is recorded in the final local summary.
This is evidence only, not a promotion target.

The only permitted follow-up is
`NEXT_UPDATE_DEPTH_POLICY_PRESERVATION_ABLATION`. Do not start C1 or switch the
formal-training default on this result.
