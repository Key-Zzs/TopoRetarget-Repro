# Penetration-loss temporal synchronization

This worktree is `integration/pene-loss-temporal-sync`, rooted at the fixed
local `main` commit `ab48770eec7d8b0750ab9f70464a55c8cdb72c24`.  It ports the
paper-external dense SDF penetration objective onto the temporal/recovery
implementation already present in that commit.  It does not modify the
protected `TopoRetarget-Repro` worktree.

The imported profile is `dense_squared_hinge_deadzone1mm_v2`: `lambda_sdf` is
0.1, the one-sided squared-hinge dead zone is 1 mm, and normalization is 1 mm.
`lambda_sdf=0` remains the E0 control and is required to be equivalent to the
main objective and temporal/checkpoint path.

The former v1 stress artifacts are pre-temporal-fix diagnostic material only.
Their audit, port inventory, excluded-hunk record, and main-fix provenance are
under `.local/reports/temporal_sync/`; replay evidence is under
`.local/experiments/s1_2b_temporal_sync_replay_v1/transport_previous_final_v2/`.

The authoritative end-to-end experiment root is
`.local/experiments/pene_loss_temporal_sync_and_stress_v2/`.  Its final report
will distinguish the legacy convex/proxy inner SDF from the original-mesh grid
inner SDF.  The grid backend is engineering validation, not a paper claim; all
final penetration audits retain `reference_winding_v1`.
