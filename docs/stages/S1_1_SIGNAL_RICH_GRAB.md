# Stage S1.1: signal-rich GRAB evaluation

## Objective

Test the frozen S1 dense squared-hinge profile on three deterministic,
penetration-active GRAB windows selected without result leakage. This stage is
a routing experiment, not a paper-method claim.

## Fixed boundaries

- Worktree: `develop/pene-loss`; all outputs are below
  `.local/experiments/s1_1_signal_rich_grab_v1/`.
- Raw GRAB, MANO, and tracked Arti-MANO assets are read-only.
- G1/G2 controls and paused G3/G4 archives are excluded from source selection.
- E0 is `lambda_sdf=0`; S1 uses only
  `dense_squared_hinge_deadzone1mm_v2` at `lambda_sdf=0.1`.
- No manual result-based reselection, mesh repair, contact-ground-truth claim,
  git mutation, or main-worktree artifact reuse.

## Dependency chain

`diagnose_g1 -> scan_source_candidates -> E0 probes -> freeze stress set ->
fast/reference audit -> full E0/S1 -> decision -> HTML smoke`

The chain is resumable. A missing active stress set fails closed before any
full S1 optimization. The formal handoff is the single
`reports/final_decision.json` plus `reports/final_summary.md` bundle.

