# Penetration-loss temporal-sync handoff

Branch: `integration/pene-loss-temporal-sync`  
Base: `ab48770eec7d8b0750ab9f70464a55c8cdb72c24`

This is an intentionally uncommitted reproduction worktree.  Do not merge,
rebase, reset, clean, stage, commit, or push it as part of experiment execution.
The protected main worktree and the older `develop/pene-loss` worktree are
read-only sources for this task.

Runtime state is resumable.  T3 grid construction writes atomic partial slabs
in the experiment-local cache and is followed by an audit/replay/T4 watcher.
Use the JSON reports under
`.local/experiments/pene_loss_temporal_sync_and_stress_v2/` as the source of
truth before resuming; never overwrite a completed artifact to restart it.

The final handoff must include protected-main integrity, temporal-fix
provenance, port manifest, backend selection, discovery coverage, frozen
selection, E0-versus-S1 evidence, determinism, HTML smoke, and a single allowed
final status.  The precise final report is generated only after all gates pass
or fail closed.
