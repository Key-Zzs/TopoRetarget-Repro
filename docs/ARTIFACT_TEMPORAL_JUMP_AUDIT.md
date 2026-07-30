# Artifact temporal-jump audit

The v1 stress artifacts were audited before this synchronization work.  The
authoritative machine-readable evidence is in
`.local/experiments/t0_artifact_jump_audit_v1/reports/`, including checkpoint
chains, per-frame robust step metrics, checkpoint-to-artifact comparisons, and
artifact-to-HTML comparisons.

The audit deliberately keeps the original v1 data immutable.  Results whose
final artifact contains a temporal discontinuity are labelled
`PRE_TEMPORAL_FIX_DIAGNOSTIC_ONLY`; they may explain the defect but cannot rank
the final stress set.  The synchronized replay starts from frame zero with
independent previous-final chains and produces new artifacts.

Main-fix provenance and the exact port boundary are recorded in
`.local/reports/temporal_sync/main_temporal_fix_provenance.json` and
`.local/reports/temporal_sync/sdf_port_manifest.json`.  The reports document
the confirmed main commits and preserve main's continuity, resume, and artifact
assembly behavior rather than importing the old implementation wholesale.
